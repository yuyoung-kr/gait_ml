from pathlib import Path
import numpy as np
import pandas as pd


def main():
    project_root = Path(__file__).resolve().parents[2]

    in_csv = project_root / "data" / "processed" / "segment_labels.csv"
    out_csv = project_root / "data" / "processed" / "segment_labels_with_targets.csv"

    if not in_csv.exists():
        raise FileNotFoundError(f"找不到输入标签文件: {in_csv}")

    df = pd.read_csv(in_csv)

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
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"segment_labels.csv 缺少字段: {missing}")

    # 初始化
    df["x"] = np.nan
    df["y"] = np.nan
    df["r"] = np.nan

    # 只对 walking 样本生成目标
    mask = (df["regression_mask"] == 1) & df["phase_pct"].notna() & (df["cycle_duration_s"] > 0)

    phi = df.loc[mask, "phase_pct"].to_numpy(dtype=float)
    t_cycle = df.loc[mask, "cycle_duration_s"].to_numpy(dtype=float)

    df.loc[mask, "x"] = np.cos(2 * np.pi * phi)
    df.loc[mask, "y"] = np.sin(2 * np.pi * phi)
    df.loc[mask, "r"] = 1.0 / t_cycle

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"已生成: {out_csv}")
    print(df.head())


if __name__ == "__main__":
    main()