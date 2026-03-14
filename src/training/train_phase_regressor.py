# 标签解释 / 回归目标定义regression_mask = 1 的 walking 样本，定义：
# φ = phase_pct ∈ [0,1)
# x = cos(2πφ) y = sin(2πφ) ， phi_current = mod(atan2(y, x) / (2*pi), 1)
#r = 1 / T_cycle ， phi_future = phi_current + r * 0.2
 
import random
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.gait_dataset import GaitDataset


# =========================
# 基本配置
# =========================
PROJECT_ROOT = Path(r"C:\Projects\gait_ml")

LABELS_FILE = PROJECT_ROOT / "data" / "processed" / "segment_labels_with_targets.csv"
SEGMENTS_DIR = PROJECT_ROOT / "data" / "segments" / "window_150ms"
SCALER_PATH = PROJECT_ROOT / "data" / "processed" / "train_scaler.pkl"

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints" / "phase_regressor"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# 被试划分
TRAIN_SUBJECTS = ["kn", "za"]
VAL_SUBJECTS = ["yy"]
TEST_SUBJECTS = ["mz"]

# 如果你决定排除某些低质量文件，可在这里加白名单/黑名单逻辑
EXCLUDE_CONDITIONS_BY_SUBJECT = {
    # 例如:
    # "mz": ["walking_4kmh", "walking_6kmh"]
}


# =========================
# 训练超参数
# =========================
SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# loss 权重
W_XY = 1.0
W_R = 0.5

GRAD_CLIP = 1.0
EARLY_STOPPING_PATIENCE = 10


# =========================
# 工具函数
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def filter_labels(dataset: GaitDataset, exclude_map: dict):
    """
    根据 subject-condition 黑名单过滤 dataset.labels
    """
    if not exclude_map:
        return

    keep_mask = []
    for _, row in dataset.labels.iterrows():
        subject = row["subject_id"]
        condition = row["condition"]
        if subject in exclude_map and condition in exclude_map[subject]:
            keep_mask.append(False)
        else:
            keep_mask.append(True)

    dataset.labels = dataset.labels.loc[keep_mask].reset_index(drop=True)


# =========================
# 模型
# 输入: (B, 8, 30)
# 输出: (B, 3) -> [x, y, r]
# =========================
class PhaseRegressorCNN(nn.Module):
    def __init__(self, in_channels=8, out_dim=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2),  # 30 -> 15

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),      # (B, 128, 1)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


# =========================
# Loss
# x,y 用 MSE
# r 用 SmoothL1，更稳
# =========================
class PhaseLoss(nn.Module):
    def __init__(self, w_xy=1.0, w_r=0.5):
        super().__init__()
        self.w_xy = w_xy
        self.w_r = w_r
        self.mse = nn.MSELoss()
        self.huber = nn.SmoothL1Loss()

    def forward(self, pred, target):
        pred_xy = pred[:, :2]
        pred_r = pred[:, 2]

        target_xy = target[:, :2]
        target_r = target[:, 2]

        loss_xy = self.mse(pred_xy, target_xy)
        loss_r = self.huber(pred_r, target_r)

        total = self.w_xy * loss_xy + self.w_r * loss_r

        return total, {
            "loss_total": float(total.detach().cpu().item()),
            "loss_xy": float(loss_xy.detach().cpu().item()),
            "loss_r": float(loss_r.detach().cpu().item()),
        }


# =========================
# 训练 / 验证
# =========================
def run_one_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_xy = 0.0
    total_r = 0.0
    n_batches = 0

    for batch in loader:
        x = batch["segment"].to(device)   # (B, 8, 30)
        y = batch["target"].to(device)    # (B, 3)

        if is_train:
            optimizer.zero_grad()

        pred = model(x)
        loss, loss_dict = criterion(pred, y)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        total_loss += loss_dict["loss_total"]
        total_xy += loss_dict["loss_xy"]
        total_r += loss_dict["loss_r"]
        n_batches += 1

    if n_batches == 0:
        raise ValueError("DataLoader 为空，请检查 dataset 是否有样本。")

    return {
        "loss_total": total_loss / n_batches,
        "loss_xy": total_xy / n_batches,
        "loss_r": total_r / n_batches,
    }


def save_checkpoint(path, epoch, model, optimizer, best_val_loss, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config,
        },
        path,
    )


# =========================
# 主函数
# =========================
def main():
    set_seed(SEED)

    print(f"DEVICE = {DEVICE}")
    print(f"LABELS_FILE = {LABELS_FILE}")
    print(f"SEGMENTS_DIR = {SEGMENTS_DIR}")

    # 1) 先建 train dataset，并 fit scaler
    train_ds = GaitDataset(
        segments_dir=SEGMENTS_DIR,
        labels_file=LABELS_FILE,
        split="train",
        train_subjects=TRAIN_SUBJECTS,
        scaler_path=SCALER_PATH,
        fit_scaler=True,
    )
    filter_labels(train_ds, EXCLUDE_CONDITIONS_BY_SUBJECT)

    # 2) val/test 读取 train scaler
    val_ds = GaitDataset(
        segments_dir=SEGMENTS_DIR,
        labels_file=LABELS_FILE,
        split="val",
        val_subjects=VAL_SUBJECTS,
        scaler_path=SCALER_PATH,
        fit_scaler=False,
    )
    filter_labels(val_ds, EXCLUDE_CONDITIONS_BY_SUBJECT)

    test_ds = GaitDataset(
        segments_dir=SEGMENTS_DIR,
        labels_file=LABELS_FILE,
        split="test",
        test_subjects=TEST_SUBJECTS,
        scaler_path=SCALER_PATH,
        fit_scaler=False,
    )
    filter_labels(test_ds, EXCLUDE_CONDITIONS_BY_SUBJECT)

    print(f"train samples = {len(train_ds)}")
    print(f"val samples   = {len(val_ds)}")
    print(f"test samples  = {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = PhaseRegressorCNN(in_channels=8, out_dim=3).to(DEVICE)
    criterion = PhaseLoss(w_xy=W_XY, w_r=W_R)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    config = {
        "labels_file": str(LABELS_FILE),
        "segments_dir": str(SEGMENTS_DIR),
        "scaler_path": str(SCALER_PATH),
        "train_subjects": TRAIN_SUBJECTS,
        "val_subjects": VAL_SUBJECTS,
        "test_subjects": TEST_SUBJECTS,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "w_xy": W_XY,
        "w_r": W_R,
        "exclude_conditions_by_subject": EXCLUDE_CONDITIONS_BY_SUBJECT,
    }

    with open(CHECKPOINT_DIR / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=DEVICE,
        )

        scheduler.step(val_metrics["loss_total"])

        current_lr = optimizer.param_groups[0]["lr"]

        log_row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss_total"],
            "train_loss_xy": train_metrics["loss_xy"],
            "train_loss_r": train_metrics["loss_r"],
            "val_loss": val_metrics["loss_total"],
            "val_loss_xy": val_metrics["loss_xy"],
            "val_loss_r": val_metrics["loss_r"],
        }
        history.append(log_row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train={train_metrics['loss_total']:.6f} "
            f"(xy={train_metrics['loss_xy']:.6f}, r={train_metrics['loss_r']:.6f}) | "
            f"val={val_metrics['loss_total']:.6f} "
            f"(xy={val_metrics['loss_xy']:.6f}, r={val_metrics['loss_r']:.6f}) | "
            f"lr={current_lr:.6e}"
        )

        # 保存 last
        save_checkpoint(
            path=CHECKPOINT_DIR / "last.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            best_val_loss=best_val_loss,
            config=config,
        )

        # 保存 best
        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            best_epoch = epoch
            patience_counter = 0

            save_checkpoint(
                path=CHECKPOINT_DIR / "best.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_val_loss=best_val_loss,
                config=config,
            )
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch = {best_epoch}")
            break

    # 保存训练历史
    import pandas as pd
    history_df = pd.DataFrame(history)
    history_df.to_csv(CHECKPOINT_DIR / "train_history.csv", index=False, encoding="utf-8-sig")

    print("\nTraining finished.")
    print(f"Best val loss = {best_val_loss:.6f} @ epoch {best_epoch}")

    # 载入 best 做 test
    ckpt = torch.load(CHECKPOINT_DIR / "best.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = run_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=DEVICE,
    )

    print(
        f"TEST  loss={test_metrics['loss_total']:.6f} "
        f"(xy={test_metrics['loss_xy']:.6f}, r={test_metrics['loss_r']:.6f})"
    )

    with open(CHECKPOINT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()