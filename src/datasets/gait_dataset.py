from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import joblib


class GaitDataset(Dataset):
    def __init__(
        self,
        segments_dir,
        labels_file,
        split="train",
        train_subjects=None,
        val_subjects=None,
        test_subjects=None,
        scaler_path=None,
        fit_scaler=False,
    ):
        self.segments_dir = Path(segments_dir)
        self.labels_file = Path(labels_file)
        self.split = split
        self.scaler_path = Path(scaler_path) if scaler_path is not None else None

        if not self.segments_dir.exists():
            raise FileNotFoundError(f"segments_dir 不存在: {self.segments_dir}")
        if not self.labels_file.exists():
            raise FileNotFoundError(f"labels_file 不存在: {self.labels_file}")

        self.labels = pd.read_csv(self.labels_file)

        required_cols = [
            "segment_id",
            "subject_id",
            "file_id",
            "trial_id",
            "activity_label",
            "condition",
            "phase_pct",
            "regression_mask",
            "cycle_duration_s",
        ]
        missing = [c for c in required_cols if c not in self.labels.columns]
        if missing:
            raise ValueError(f"labels_file 缺少字段: {missing}")

        # 只保留 walking 数据做 phase regression
        self.labels = self.labels[self.labels["regression_mask"] == 1].reset_index(drop=True)

        if split == "train":
            if train_subjects is None:
                raise ValueError("train split 需要提供 train_subjects")
            self.labels = self.labels[self.labels["subject_id"].isin(train_subjects)].reset_index(drop=True)

        elif split == "val":
            if val_subjects is None:
                raise ValueError("val split 需要提供 val_subjects")
            self.labels = self.labels[self.labels["subject_id"].isin(val_subjects)].reset_index(drop=True)

        elif split == "test":
            if test_subjects is None:
                raise ValueError("test split 需要提供 test_subjects")
            self.labels = self.labels[self.labels["subject_id"].isin(test_subjects)].reset_index(drop=True)

        else:
            raise ValueError("split 必须是 train / val / test")

        if len(self.labels) == 0:
            raise ValueError(f"{split} split 没有样本，请检查 subject 划分")

        self.scaler = StandardScaler()

        if fit_scaler:
            self._fit_scaler()
        else:
            self._load_scaler()

    def _build_npy_path(self, row):
        filename = f"{row['subject_id']}_{row['file_id']}_{row['condition']}_{row['trial_id']}_segments.npy"
        return self.segments_dir / filename

    def _fit_scaler(self):
        if self.scaler_path is None:
            raise ValueError("fit_scaler=True 时必须提供 scaler_path")

        all_rows = []

        for _, row in self.labels.iterrows():
            npy_path = self._build_npy_path(row)
            if not npy_path.exists():
                continue

            arr = np.load(npy_path)  # shape: (N, 30, 8)
            seg_idx = int(row["segment_id"])

            if seg_idx < 0 or seg_idx >= len(arr):
                continue

            seg = arr[seg_idx]  # (30, 8)
            all_rows.append(seg)

        if not all_rows:
            raise ValueError("没有可用于 fit scaler 的训练样本")

        all_rows = np.concatenate(all_rows, axis=0)  # (总时间点数, 8)
        self.scaler.fit(all_rows)

        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)

    def _load_scaler(self):
        if self.scaler_path is None or not self.scaler_path.exists():
            raise FileNotFoundError(
                f"找不到 scaler 文件: {self.scaler_path}。\n"
                "请先创建 train dataset，并设置 fit_scaler=True。"
            )

        self.scaler = joblib.load(self.scaler_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]

        npy_path = self._build_npy_path(row)
        if not npy_path.exists():
            raise FileNotFoundError(f"找不到 segment 文件: {npy_path}")

        arr = np.load(npy_path)  # (N, 30, 8)
        seg_idx = int(row["segment_id"])

        if seg_idx < 0 or seg_idx >= len(arr):
            raise IndexError(f"segment_id 越界: {seg_idx}, 文件: {npy_path.name}, 总段数: {len(arr)}")

        segment = arr[seg_idx]  # (30, 8)
        segment = self.scaler.transform(segment)
        segment = torch.tensor(segment, dtype=torch.float32).permute(1, 0)  # (8, 30)

        phi = float(row["phase_pct"])
        x = np.cos(2 * np.pi * phi)
        y = np.sin(2 * np.pi * phi)

        cycle_duration = float(row["cycle_duration_s"])
        r = 1.0 / cycle_duration if cycle_duration > 0 else 0.0

        target = torch.tensor([x, y, r], dtype=torch.float32)

        return {
            "segment": segment,      # (8, 30)
            "target": target,        # (3,)
            "condition": row["condition"],
            "subject_id": row["subject_id"],
            "file_id": row["file_id"],
            "trial_id": row["trial_id"],
            "segment_id": int(row["segment_id"]),
            "phase_pct": torch.tensor(phi, dtype=torch.float32),
        }