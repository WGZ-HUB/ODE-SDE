# -*- coding: utf-8 -*-
"""
STFT 2D U-Net baseline for complex radar anti-jamming.

Dataset directory:
    data_dir/
        echo.mat   variable name: echo, shape [N, L], complex
        sig.mat    variable name: sig,  shape [N, L], complex

Tasks:
    1) Train a supervised 2D U-Net in STFT domain:
           echo -> STFT(real, imag) -> U-Net -> clean STFT(real, imag)
    2) Inference and save time-domain complex output:
           unet_stft2d_result.mat, variable name: result

Recommended usage:

Train:
python unet_stft2d_baseline.py ^
  --mode train ^
  --data_dir "E:/GenRadar/data/ISRJ" ^
  --save_dir "E:/GenRadar/checkpoints/unet_isrj" ^
  --epochs 80 ^
  --batch_size 16

Inference:
python unet_stft2d_baseline.py ^
  --mode infer ^
  --data_dir "E:/GenRadar/data/ISRJ" ^
  --save_dir "E:/GenRadar/checkpoints/unet_isrj" ^
  --ckpt "E:/GenRadar/checkpoints/unet_isrj/best_unet.pt"
"""

import os
import argparse
import random
from pathlib import Path

import numpy as np
import scipy.io as sio
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import time


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mat_var(path: str, var_name: str):
    mat = sio.loadmat(path)
    if var_name not in mat:
        keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"{path} 中找不到变量 {var_name}，可用变量有: {keys}")
    return mat[var_name]


def to_complex_numpy(x):
    x = np.asarray(x)
    if np.iscomplexobj(x):
        return x.astype(np.complex64)
    return x.astype(np.float32).astype(np.complex64)


def complex_stft_to_2ch(wave, n_fft=256, hop_length=64, win_length=256):
    """
    wave: complex tensor, shape [L]
    return: float tensor, shape [2, F, T], F=n_fft, because onesided=False
    """
    if not torch.is_complex(wave):
        wave = wave.to(torch.complex64)

    window = torch.hann_window(win_length, device=wave.device)
    spec = torch.stft(
        wave,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
        onesided=False,
    )
    return torch.stack([spec.real, spec.imag], dim=0).float()


def spec_2ch_to_complex_wave(spec_2ch, length, n_fft=256, hop_length=64, win_length=256):
    """
    spec_2ch: float tensor, shape [2, F, T]
    return: complex tensor, shape [L]
    """
    spec = torch.complex(spec_2ch[0], spec_2ch[1])
    window = torch.hann_window(win_length, device=spec.device)
    wave = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        onesided=False,
        length=length,
        return_complex=True,
    )
    return wave


class RadarSTFTDataset(Dataset):
    """
    input  = STFT(echo)
    target = STFT(sig)
    Each sample is normalized by max(abs(echo)). The scale is returned for inference.
    """

    def __init__(self, data_dir, max_samples=-1, n_fft=256, hop_length=64, win_length=256):
        super().__init__()
        echo_path = os.path.join(data_dir, "echo.mat")
        sig_path = os.path.join(data_dir, "sig.mat")

        self.echo = to_complex_numpy(load_mat_var(echo_path, "echo"))
        self.sig = to_complex_numpy(load_mat_var(sig_path, "sig"))

        if self.echo.shape != self.sig.shape:
            raise ValueError(f"echo 和 sig 形状不一致: echo={self.echo.shape}, sig={self.sig.shape}")
        if self.echo.ndim != 2:
            raise ValueError(f"echo/sig 应该是 [N, L]，当前 echo shape={self.echo.shape}")

        if max_samples > 0:
            self.echo = self.echo[:max_samples]
            self.sig = self.sig[:max_samples]

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.length = self.echo.shape[1]

    def __len__(self):
        return self.echo.shape[0]

    def __getitem__(self, idx):
        echo = torch.from_numpy(self.echo[idx]).to(torch.complex64)
        sig = torch.from_numpy(self.sig[idx]).to(torch.complex64)

        scale = torch.max(torch.abs(echo))
        if scale < 1e-8:
            scale = torch.tensor(1.0, dtype=torch.float32)

        echo_norm = echo / scale
        sig_norm = sig / scale

        x = complex_stft_to_2ch(echo_norm, self.n_fft, self.hop_length, self.win_length)
        y = complex_stft_to_2ch(sig_norm, self.n_fft, self.hop_length, self.win_length)

        return {"input": x, "target": y, "scale": scale.float(), "idx": idx}


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch, dropout))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class STFTUNet2D(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, base_ch=32, dropout=0.0, residual=True):
        super().__init__()
        self.residual = residual
        self.inc = DoubleConv(in_ch, base_ch, dropout)
        self.down1 = Down(base_ch, base_ch * 2, dropout)
        self.down2 = Down(base_ch * 2, base_ch * 4, dropout)
        self.down3 = Down(base_ch * 4, base_ch * 8, dropout)
        self.up1 = Up(base_ch * 8, base_ch * 4, base_ch * 4, dropout)
        self.up2 = Up(base_ch * 4, base_ch * 2, base_ch * 2, dropout)
        self.up3 = Up(base_ch * 2, base_ch, base_ch, dropout)
        self.outc = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        y = self.up1(x4, x3)
        y = self.up2(y, x2)
        y = self.up3(y, x1)
        y = self.outc(y)
        if self.residual:
            y = x + y
        return y


def complex_spectral_mse(pred, target):
    return F.mse_loss(pred, target)


def complex_mag_loss(pred, target):
    pred_mag = torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2 + 1e-8)
    target_mag = torch.sqrt(target[:, 0] ** 2 + target[:, 1] ** 2 + 1e-8)
    return F.l1_loss(pred_mag, target_mag)


def total_loss(pred, target, mag_weight=0.2):
    return complex_spectral_mse(pred, target) + mag_weight * complex_mag_loss(pred, target)


@torch.no_grad()
def validate(model, loader, device, mag_weight=0.2):
    model.eval()
    losses = []
    for batch in loader:
        x = batch["input"].to(device)
        y = batch["target"].to(device)
        pred = model(x)
        losses.append(total_loss(pred, y, mag_weight).item())
    return float(np.mean(losses)) if losses else np.inf


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    dataset = RadarSTFTDataset(args.data_dir, args.max_samples, args.nfft, args.hop_length, args.win_length)
    n_total = len(dataset)
    n_train = int(n_total * args.train_ratio)
    n_val = n_total - n_train

    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = STFTUNet2D(2, 2, args.base_ch, args.dropout, residual=(not args.no_residual)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.05)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best_unet.pt"
    last_path = save_dir / "last_unet.pt"
    best_val = np.inf

    print("========== STFT 2D U-Net Training ==========")
    print(f"data_dir   : {args.data_dir}")
    print(f"samples    : total={n_total}, train={n_train}, val={n_val}")
    print(f"nfft/hop   : {args.nfft}/{args.hop_length}")
    print(f"batch_size : {args.batch_size}")
    print(f"epochs     : {args.epochs}")
    print(f"base_ch    : {args.base_ch}")
    print(f"save_dir   : {args.save_dir}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for batch in pbar:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            pred = model(x)
            loss = total_loss(pred, y, args.mag_weight)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4e}")

        if scheduler is not None:
            scheduler.step()

        train_loss = float(np.mean(train_losses))
        val_loss = validate(model, val_loader, device, args.mag_weight)
        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6e}, val_loss={val_loss:.6e}")

        ckpt = {"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val_loss": val_loss}
        torch.save(ckpt, last_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, best_path)
            print(f"  ✅ Saved best model: {best_path}, val_loss={best_val:.6e}")

    print(f"Training done. Best checkpoint: {best_path}")


@torch.no_grad()
def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    ckpt_path = args.ckpt or os.path.join(args.save_dir, "best_unet.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    train_args = ckpt.get("args", {})

    base_ch = int(train_args.get("base_ch", args.base_ch))
    dropout = float(train_args.get("dropout", args.dropout))
    residual = not bool(train_args.get("no_residual", args.no_residual))

    model = STFTUNet2D(2, 2, base_ch, dropout, residual=residual).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n========== Model Parameters ==========")
    print(f"Total params     : {num_params / 1e6:.4f} M")
    print(f"Trainable params : {num_trainable_params / 1e6:.4f} M")

    dataset = RadarSTFTDataset(args.data_dir, args.max_samples, args.nfft, args.hop_length, args.win_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    N = len(dataset)
    L = dataset.length
    result = np.zeros((N, L), dtype=np.complex64)

    print("========== STFT 2D U-Net Inference ==========")
    print(f"data_dir : {args.data_dir}")
    print(f"ckpt     : {ckpt_path}")
    print(f"samples  : {N}")
    print(f"output   : {os.path.join(args.data_dir, args.output_name)}")

    write_pos = 0
    inference_times = []
    for batch in tqdm(loader, desc="Infer", unit="batch"):
        x = batch["input"].to(device)
        scales = batch["scale"].to(device)

        # ================== Start timing ==================
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        pred_spec = model(x)
        bsz = x.shape[0]
        for b in range(bsz):
            wave = spec_2ch_to_complex_wave(pred_spec[b], L, args.nfft, args.hop_length, args.win_length)
            wave = wave * scales[b]
            result[write_pos + b] = wave.detach().cpu().numpy().astype(np.complex64)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        # ================== End timing ==================

        batch_time_ms = (end_time - start_time) * 1000
        inference_times.append(batch_time_ms / bsz)  # ms per pulse/sample

        write_pos += bsz

    # out_path = os.path.join(args.data_dir, args.output_name)
    # sio.savemat(out_path, {"ex_sig": result})
    # print(f"✅ Inference done, saved: {out_path}")
    # print(f"result shape: {result.shape}")

    inference_times = np.array(inference_times)

    # Remove the first batch to avoid warm-up overhead
    valid_times = inference_times[0:] if len(inference_times) > 1 else inference_times

    print("\n========== Inference Time Statistics ==========")
    print(f"Samples counted : {len(valid_times)} batches")
    print(f"Mean time       : {np.mean(valid_times):.4f} ms / pulse")
    print(f"Std time        : {np.std(valid_times):.4f} ms")
    print(f"Min time        : {np.min(valid_times):.4f} ms")
    print(f"Max time        : {np.max(valid_times):.4f} ms")


def build_parser():
    parser = argparse.ArgumentParser("STFT 2D U-Net radar anti-jamming baseline")
    parser.add_argument("--mode", default='infer')        # choices=["train", "infer"]
    parser.add_argument("--data_dir", default='E:\daima\MeanFlow\data\ISRJ_CSNJ\JSR\SNR0_JSR0\\', help="包含 echo.mat 和 sig.mat 的数据文件夹")
    # parser.add_argument("--save_dir", default='E:/daima/MeanFlow/baseline/UNet/ISRJ_CSNJ', help="模型保存文件夹")
    parser.add_argument("--ckpt", default="E:/daima/MeanFlow/baseline/UNet/ISRJ_CSNJ/best_unet.pt", help="推理时加载的 checkpoint")
    parser.add_argument("--output_name", default="ex_sig_unet.mat")

    parser.add_argument("--max_samples", type=int, default=-1, help="-1 表示使用全部样本")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--nfft", type=int, default=256)
    parser.add_argument("--hop_length", type=int, default=64)
    parser.add_argument("--win_length", type=int, default=256)

    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no_residual", action="store_true")

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mag_weight", type=float, default=0.2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    return parser


def main():
    args = build_parser().parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "infer":
        infer(args)


if __name__ == "__main__":
    main()

    
