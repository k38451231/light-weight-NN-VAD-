# 基本套件
import os, math, glob, json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import librosa
import librosa.display

SR = 16000
N_MELS = 40
N_FFT = 512
WIN_LENGTH = 400     # 25 ms at 16k
HOP_LENGTH = 160     # 10 ms at 16k  → lab_frame 的時間解析度
EPS = 1e-8


DATA_ROOT = Path("D:\\VAD_project\\data\\training_data") 
assert DATA_ROOT.exists(), f"找不到資料夾：{DATA_ROOT.resolve()}"


def pad_frame(wav: np.ndarray, frame_len: int, hop: int, target_frames: int) -> np.ndarray:
    #STFT(center=False) frames = 1 + floor((N + pad_end - frame_len)/hop) 回推
    N = len(wav)
    pad_end = (target_frames - 1) * hop + frame_len - N
    if pad_end > 0:
        wav = np.pad(wav, (0, pad_end), mode="constant")
    return wav

def wav_to_logmel(wav: np.ndarray, sr=SR, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH, n_mels=N_MELS, target_frames=None):
    #return shape: [n_mels, T]
    if target_frames is not None:
        wav = pad_frame(wav, N_FFT, hop_length, target_frames)

    # STFT 用 center=False幀數 = 1 + floor((N - win)/hop)
    S = librosa.feature.melspectrogram(
        y=wav.astype(np.float32),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        center=False, #不要置中
        power=2.0,
    )
    logmel = librosa.power_to_db(S + EPS, ref=np.max) # power轉乘dB scale

    # 對每個貧帶normalize # [nmel, T] = [40, T]
    logmel = (logmel - logmel.mean(axis=1, keepdims=True)) / (logmel.std(axis=1, keepdims=True) + 1e-5)
    return logmel

class VADDataset(Dataset):
    def __init__(self, root: Path, split: str = "train"):

        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split / "result"
        assert self.split_dir.exists(), f"找不到 split 目錄：{self.split_dir}"

        # collect folder(每筆音檔跟label都有獨立資料夾)
        self.samples = sorted([p for p in self.split_dir.iterdir() if p.is_dir()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample_dir = self.samples[idx]
        lab_frame = np.load(sample_dir / "lab_frame.npy")

        arr = np.load(sample_dir / "audio.npy", allow_pickle=True)
        noise_wav, noisy_wav, clean_wav = arr[0], arr[1], arr[2]

        T = int(lab_frame.shape[0]) # lab_frame (400,)
        x = noisy_wav[0]
        N = len(x)

        # log-mel
        logmel = wav_to_logmel(x, sr=SR, n_fft=N_FFT, win_length=WIN_LENGTH,
                                  hop_length=HOP_LENGTH, n_mels=N_MELS, target_frames=T)  # (40, T)
        assert logmel.shape[1] == T, f"logmel幀數({logmel.shape[1]}) label幀數({T}) 不同：{sample_dir}"

        # torch tensor
        feat = torch.from_numpy(logmel).float()         # [40, T]
        labf = torch.from_numpy(lab_frame.astype(np.int64))  # [T], 0/1

        #return dict
        return {
            "feat": feat,              # [40, T]
            "label": labf,             # [T]
            "wav_len": torch.tensor(N, dtype=torch.int32),
            "path": str(sample_dir),
        }

def make_loader(root=DATA_ROOT, split="train", batch_size=8, shuffle=True, num_workers=0):
    ds = VADDataset(root, split)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), #讓GPU直接從 "固定的" memory位置讀資料 => 加速
    )
    return ds, dl

# test
train_ds, train_dl = make_loader(DATA_ROOT, "train", batch_size=4, shuffle=True, num_workers=0)
valid_ds, valid_dl = make_loader(DATA_ROOT, "valid", batch_size=4, shuffle=False, num_workers=0)
test_ds,  test_dl  = make_loader(DATA_ROOT, "test" , batch_size=4, shuffle=False, num_workers=0)

print(f"#train = {len(train_ds)}, #valid = {len(valid_ds)}, #test = {len(test_ds)}")


batch = next(iter(train_dl)) #iter(train_dl) -> create一個iterator 拿一個 batch 看看
x = batch["feat"]     # [B, 40, T]
y = batch["label"]    # [B, T]
paths = batch["path"]

print("feat shape:", x.shape) #4, 40, 400
print("label shape:", y.shape) # 4, 400
print("paths[0]:", paths[0])



