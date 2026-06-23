import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio
import h5py
import os
from tqdm import tqdm
import time

# ==========================================
# 1. 参数初始化
# ==========================================
Tr = 40e-6              # 脉冲重复周期
B = 40e6                # 瞬时带宽
fs = 4 * B              # 采样频率 160 MHz
Tp = 10e-6              # 脉冲宽度
f0 = 10e9               # 载频

K = B / Tp              # 调频斜率
nrn = int(fs * Tr)      # 单个脉冲采样点数
t_fast = np.arange(nrn) / fs

# ==========================================
# 2. 文件路径
# ==========================================
data_path = 'E:\daima\MeanFlow\data\ISRJ_CSNJ\JSR\SNR0_JSR40\\'

echo_file = os.path.join(data_path, 'echo.mat')
sig_file  = os.path.join(data_path, 'sig.mat')
jam_file  = os.path.join(data_path, 'jam.mat')

save_file = os.path.join(data_path, 'ex_sig_stft_dy.mat')

# ==========================================
# 3. 读取 mat 文件函数
# ==========================================
def load_mat_variable(file_path, var_name):
    """
    自动读取普通 mat
    返回 numpy 数组
    """
    data = sio.loadmat(file_path)
    if var_name not in data:
        raise KeyError(f"{file_path} 中找不到变量 {var_name}")
    arr = data[var_name]
    return np.asarray(arr)

# ==========================================
# 4. 读取数据
# ==========================================
echo = load_mat_variable(echo_file, 'echo')
sig  = load_mat_variable(sig_file,  'sig')
jam  = load_mat_variable(jam_file,  'jam')

echo = np.asarray(echo)
sig  = np.asarray(sig)
jam  = np.asarray(jam)

# 如果是一维数据，转成二维
if echo.ndim == 1:
    echo = echo.reshape(1, -1)

if sig.ndim == 1:
    sig = sig.reshape(1, -1)

if jam.ndim == 1:
    jam = jam.reshape(1, -1)

num_rows, num_cols = echo.shape

print("========== 数据读取完成 ==========")
print(f"echo 形状: {echo.shape}")
print(f"sig  形状: {sig.shape}")
print(f"jam  形状: {jam.shape}")
print(f"自动检测到共有 {num_rows} 条回波，每条 {num_cols} 个采样点")

# 更新 nrn，防止和数据真实长度不一致
nrn = num_cols
t_fast = np.arange(nrn) / fs

# ==========================================
# 5. STFT 自适应时频掩膜函数
# ==========================================
def stft_adaptive_mask_filter(
    echo_pulse,
    fs,
    B,
    Tp,
    delay=20e-6,
    nperseg=128,
    noverlap=120,
    search_bw=15e6,
    edge_ratio=0.25,
    mean_factor=1.20,
    min_bw=2e6,
    max_bw=12e6
):
    """
    对单条回波进行 STFT 自适应时频掩膜抗干扰
    """
    K = B / Tp
    echo_pulse = np.asarray(echo_pulse).squeeze()
    nrn = len(echo_pulse)

    # ---------- STFT ----------
    f_stft, t_stft, Zxx = signal.stft(
        echo_pulse,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False
    )

    f_stft = np.fft.fftshift(f_stft)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    df = np.abs(f_stft[1] - f_stft[0])

    # ---------- 构造自适应 mask ----------
    mask = np.zeros_like(Zxx, dtype=bool)

    for i, t_val in enumerate(t_stft):

        # 只处理目标可能存在的时间段
        if not (delay <= t_val <= delay + Tp):
            continue

        # 默认 LFM 基带轨迹为 -B/2 到 +B/2
        target_f = K * (t_val - delay) - B / 2
        target_f = np.clip(target_f, -B / 2, B / 2)

        center_idx = np.argmin(np.abs(f_stft - target_f))

        half_search_bins = int((search_bw / 2) / df)
        left_min = max(0, center_idx - half_search_bins)
        right_max = min(len(f_stft) - 1, center_idx + half_search_bins)

        local_mag = np.abs(Zxx[:, i])
        search_mag = local_mag[left_min:right_max + 1]

        if len(search_mag) < 3:
            continue

        peak_val = np.max(search_mag)
        mean_val = np.mean(search_mag)

        adaptive_thresh = max(edge_ratio * peak_val, mean_factor * mean_val)

        # 向左搜索边界
        left_idx = center_idx
        while left_idx > left_min:
            if local_mag[left_idx] < adaptive_thresh and local_mag[left_idx] <= local_mag[left_idx + 1]:
                break
            left_idx -= 1

        # 向右搜索边界
        right_idx = center_idx
        while right_idx < right_max:
            if local_mag[right_idx] < adaptive_thresh and local_mag[right_idx] <= local_mag[right_idx - 1]:
                break
            right_idx += 1

        # 限制通带宽度
        current_bw = (right_idx - left_idx + 1) * df

        if current_bw < min_bw:
            half_bins = int((min_bw / 2) / df)
            left_idx = max(0, center_idx - half_bins)
            right_idx = min(len(f_stft) - 1, center_idx + half_bins)

        elif current_bw > max_bw:
            half_bins = int((max_bw / 2) / df)
            left_idx = max(0, center_idx - half_bins)
            right_idx = min(len(f_stft) - 1, center_idx + half_bins)

        mask[left_idx:right_idx + 1, i] = True

    # ---------- ISTFT ----------
    Zxx_filtered = Zxx * mask
    Zxx_filtered_ishift = np.fft.ifftshift(Zxx_filtered, axes=0)

    _, rec_pulse = signal.istft(
        Zxx_filtered_ishift,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        input_onesided=False
    )

    # 保证长度一致
    rec_pulse = rec_pulse[:nrn]

    if len(rec_pulse) < nrn:
        rec_pulse = np.pad(rec_pulse, (0, nrn - len(rec_pulse)), mode='constant')

    return rec_pulse

# ==========================================
# 6. 批量抗干扰
# ==========================================
rec_echo = np.zeros_like(echo, dtype=np.complex128)

print("\n========== 开始批量 STFT 自适应时频掩膜抗干扰 ==========")

inference_times = []

for idx in tqdm(range(num_rows), desc="Processing"):
    echo_pulse = echo[idx, :]

    # ================== 开始计时 ==================
    start_time = time.perf_counter()

    rec_pulse = stft_adaptive_mask_filter(
        echo_pulse=echo_pulse,
        fs=fs,
        B=B,
        Tp=Tp,
        delay=20e-6,
        nperseg=128,
        noverlap=120,
        search_bw=15e6,
        edge_ratio=0.25,
        mean_factor=1.20,
        min_bw=2e6,
        max_bw=12e6
    )

    rec_echo[idx, :] = rec_pulse

    # ================== 结束计时 ==================
    end_time = time.perf_counter()
    inference_times.append((end_time - start_time) * 1000)

print("========== 批量抗干扰完成 ==========")

# # ==========================================
# # 7. 保存 mat 文件
# # ==========================================
# sio.savemat(
#     save_file,
#     {
#         'ex_sig': rec_echo,
#     }
# )

# print(f"结果已保存到: {save_file}")
# print(f"保存变量名: ex_sig")
# print(f"ex_sig 形状: {rec_echo.shape}")

# # ==========================================
# # 8. 可选：画一条样本检查效果
# # ==========================================
# check_idx = 0

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# plt.figure(figsize=(12, 4))
# plt.plot(t_fast * 1e6, np.real(echo[check_idx, :]), label='受干扰回波', alpha=0.5)
# plt.plot(t_fast * 1e6, np.real(sig[check_idx, :]), label='纯目标信号', linewidth=2)
# plt.plot(t_fast * 1e6, np.real(rec_echo[check_idx, :]), label='STFT自适应抗干扰后', linestyle='--')
# plt.xlabel('时间 / μs')
# plt.ylabel('幅度')
# plt.title(f'第 {check_idx} 条回波抗干扰效果检查')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# # plt.show()




inference_times = np.array(inference_times)

# 去掉第一个样本，避免首次调用带来的 warm-up 影响
valid_times = inference_times[1:] if len(inference_times) > 1 else inference_times

print("\n========== Inference Time Statistics ==========")
print(f"Samples counted : {len(valid_times)} pulses")
print(f"Mean time       : {np.mean(valid_times):.4f} ms / pulse")
print(f"Std time        : {np.std(valid_times):.4f} ms")
print(f"Min time        : {np.min(valid_times):.4f} ms")
print(f"Max time        : {np.max(valid_times):.4f} ms")
print(f"Total time      : {np.sum(valid_times):.4f} ms")