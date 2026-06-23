import numpy as np
from scipy import signal
import scipy.io as sio
import h5py
import os
import time

# ==========================================
# 1. 参数初始化
# ==========================================
Tr = 40e-6              # 脉冲重复周期
PRF = 1 / Tr            # 脉冲重复频率
B = 40e6                # 瞬时带宽
fs = 4 * B              # 采样频率 (160MHz)
Tp = 10e-6              # 脉冲宽度
pul_num = 64            # 脉冲个数
f0 = 10e9               # 载频

K = B / Tp              # 调频斜率
nrn = int(fs * Tr)      # 单个脉冲的采样点数 (6400)

# ==========================================
# 2. 读取所有回波数据并自动识别行数
# ==========================================
data_path = 'E:\daima\MeanFlow\data\ISRJ_CSNJ\JSR\SNR0_JSR0\\'

print("正在加载数据，请稍候...")
try:
    # 常规读取
    echo_data = sio.loadmat(data_path + 'echo.mat')
    echo_matrix = echo_data['echo']  # 期望维度: (N_pulses, 6400)
    
except NotImplementedError:
    # HDF5 读取 -v7.3 格式
    print("检测到 -v7.3 格式，切换使用 h5py 读取...")
    with h5py.File(data_path + 'echo.mat', 'r') as f_echo:
        # 如果 MATLAB 中是 (N, 6400)，读出来是 (6400, N)，所以需要 .T 转置回来
        echo_raw = f_echo['echo'][:]
        echo_matrix = (echo_raw['real'] + 1j * echo_raw['imag']).T

num_pulses = echo_matrix.shape[0]
print(f"成功读取数据！共检测到 {num_pulses} 行 (脉冲) 数据，长度为 {echo_matrix.shape[1]}。")

# 创建一个全零的复数矩阵，用于保存抗干扰恢复后的数据
rec_matrix = np.zeros_like(echo_matrix, dtype=complex)

# ==========================================
# 3. 预先计算 STFT 掩码 (Mask) - 性能优化的关键！
# ==========================================
nperseg = 128  # 窗长
noverlap = 120 # 重叠点数
window_bw = 8e6  # 8MHz 的容差窗
delay = 20e-6    # 目标的起始到达时间

print("正在预计算滤波掩码...")
# 用一个 Dummy 数组走一遍 STFT，只为了获取 f_stft 和 t_stft 坐标网格
dummy_pulse = np.zeros(nrn)
f_stft, t_stft, _ = signal.stft(dummy_pulse, fs=fs, window='hann', 
                                nperseg=nperseg, noverlap=noverlap, return_onesided=False)
f_stft = np.fft.fftshift(f_stft)

# 提前生成 Mask (所有脉冲共用这个 Mask)
mask = np.zeros((len(f_stft), len(t_stft)), dtype=bool)
for i, t_val in enumerate(t_stft):
    if delay <= t_val <= delay + Tp:
        target_f = K * (t_val - delay) - B/2
        target_f = np.clip(target_f, -B/2, B/2)
        mask[:, i] = np.abs(f_stft - target_f) < (window_bw / 2)

# ==========================================
# 4. 批量进行传统 STFT 抗干扰处理
# ==========================================
print("开始进行批量抗干扰处理...")
inference_times = []

for idx in range(num_pulses):
    # 每处理 100 行打印一次进度，避免频繁打印拖慢速度
    if idx % 100 == 0:
        print(f"进度: 正在处理第 {idx} / {num_pulses} 个脉冲...")
        
    echo_pulse = echo_matrix[idx, :]

    # ================== 开始计时 ==================
    start_time = time.perf_counter()
    
    # 步骤 A: 计算 STFT
    _, _, Zxx = signal.stft(echo_pulse, fs=fs, window='hann', 
                            nperseg=nperseg, noverlap=noverlap, return_onesided=False)
    Zxx = np.fft.fftshift(Zxx, axes=0)
    
    # 步骤 B: 应用预先计算好的掩码
    Zxx_filtered = Zxx * mask
    
    # 步骤 C: 逆变换 (ISTFT)
    Zxx_filtered_ishift = np.fft.ifftshift(Zxx_filtered, axes=0)
    _, rec_pulse = signal.istft(Zxx_filtered_ishift, fs=fs, window='hann', 
                                nperseg=nperseg, noverlap=noverlap, input_onesided=False)
    
    
    # ================== 结束计时 ==================
    end_time = time.perf_counter()
    inference_times.append((end_time - start_time) * 1000)
    
    # 确保存入结果矩阵的长度为 nrn
    rec_matrix[idx, :] = rec_pulse[:nrn]

print("100% - 所有脉冲处理完成！")


# ==========================================
# 5. 将处理后的数据保存为新的 .mat 文件
# ==========================================
# save_name = 'ex_sig_stft_fix.mat'
# save_path = os.path.join(data_path, save_name)

# # 保存文件，字典的 Key 'rec_echo' 就是你在 MATLAB 里加载后看到的变量名
# sio.savemat(save_path, {'ex_sig': rec_matrix})

# print(f"✅ 抗干扰后的数据已成功保存至：{save_path}")
# print("你现在可以在 MATLAB 中加载 'echo_filtered_stft.mat' 并在变量 'rec_echo' 中查看结果。")



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