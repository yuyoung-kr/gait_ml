from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEGMENTS_DIR = PROJECT_ROOT / "data" / "segments" / "window_150ms"
RAW_ALIGNED_DIR = PROJECT_ROOT / "data" / "raw_aligned"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "segment_labels.csv"


def load_all_segment_label_files():
    files = sorted(SEGMENTS_DIR.glob("*_segment_labels.csv"))
    if not files:
        raise FileNotFoundError(
            f"找不到任何 segment label 文件:\n{SEGMENTS_DIR}\n"
            "请先确认 preprocess 是否已经生成 *_segment_labels.csv"
        )

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_label_file"] = f.name
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    return merged


def load_all_raw_aligned():
    files = sorted(RAW_ALIGNED_DIR.glob("*_aligned.csv"))
    if not files:
        raise FileNotFoundError(
            f"找不到任何 raw_aligned 文件:\n{RAW_ALIGNED_DIR}"
        )

    dfs = []
    for f in files:
        df = pd.read_csv(f)

        needed = [
            "subject_id",
            "file_id",
            "trial_id",
            "condition",
            "timestamp",
            "cycle_duration_s",
            "phase_pct",
            "phase_valid_mask",
            "steady_mask",
            "activity_label",
        ]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"{f.name} 缺少字段: {missing}")

        dfs.append(df[needed].copy())

    merged = pd.concat(dfs, ignore_index=True)
    return merged


def merge_labels_with_aligned(seg_df, aligned_df):
    # 为避免浮点时间戳匹配误差，先统一 round
    seg_df = seg_df.copy()
    aligned_df = aligned_df.copy()

    seg_df["segment_end_time_round"] = seg_df["segment_end_time"].round(6)
    aligned_df["timestamp_round"] = aligned_df["timestamp"].round(6)

    merged = seg_df.merge(
        aligned_df,
        left_on=["subject_id", "file_id", "trial_id", "condition", "segment_end_time_round"],
        right_on=["subject_id", "file_id", "trial_id", "condition", "timestamp_round"],
        how="left",
        suffixes=("", "_aligned"),
    )

    # 如果 segment 表里的 phase_pct 和 aligned 里的 phase_pct 不一致，以 segment 为主
    # 但 cycle_duration_s 必须从 aligned 补
    if "cycle_duration_s" not in merged.columns:
        merged["cycle_duration_s"] = merged["cycle_duration_s_aligned"]

    # 有些 merge 后会出现重复列，统一清理
    drop_cols = [c for c in merged.columns if c.endswith("_aligned")]
    drop_cols += ["segment_end_time_round", "timestamp_round", "timestamp"]
    drop_cols = [c for c in drop_cols if c in merged.columns]
    merged = merged.drop(columns=drop_cols)

    return merged


def validate_output(df):
    required_cols = [
        "segment_id",
        "subject_id",
        "file_id",
        "trial_id",
        "activity_label",
        "condition",
        "segment_start_time",
        "segment_end_time",
        "window_ms",
        "phase_pct",
        "regression_mask",
        "steady_mask",
        "cycle_duration_s",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"最终 segment_labels.csv 缺少字段: {missing}")


def main():
    print("1) 读取单文件 segment labels ...")
    seg_df = load_all_segment_label_files()
    print(f"   找到 {len(seg_df)} 条 segment label")

    print("2) 读取 raw_aligned ...")
    aligned_df = load_all_raw_aligned()
    print(f"   找到 {len(aligned_df)} 条 aligned 行")

    print("3) 合并并补充 cycle_duration_s ...")
    merged = merge_labels_with_aligned(seg_df, aligned_df)

    print("4) 校验字段 ...")
    validate_output(merged)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n✅ 已生成: {OUTPUT_CSV}")
    print(merged.head())


if __name__ == "__main__":
    main()