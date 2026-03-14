from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===== 可改参数 =====
WINDOW_SEC = 8.0
USE_SUSPICIOUS_ONLY = True

# 如果你只想先画这两个，可直接写：
MANUAL_TARGETS = [
    ("kn", "walking_6kmh"),
    ("mz", "walking_6kmh"),
]
# 如果想全部 suspicious 文件都画，保留上面这个列表也没关系，
# 但 USE_SUSPICIOUS_ONLY=True 时会优先用 gait_event_summary_suspicious.csv


def find_raw_aligned_file(raw_aligned_dir: Path, subject_id: str, file_id: str, condition: str) -> Path | None:
    exact_candidates = list(raw_aligned_dir.glob(f"*{file_id}*.csv"))
    if exact_candidates:
        return exact_candidates[0]

    fuzzy_candidates = list(raw_aligned_dir.glob(f"*{subject_id}*{condition}*.csv"))
    if fuzzy_candidates:
        return fuzzy_candidates[0]

    return None


def find_gait_event_file(gait_events_dir: Path, subject_id: str, file_id: str, condition: str) -> Path | None:
    cond_dir = gait_events_dir / condition
    if not cond_dir.exists():
        return None

    exact = cond_dir / f"{subject_id}_{file_id}_{condition}_gait_events.csv"
    if exact.exists():
        return exact

    fuzzy = list(cond_dir.glob(f"*{subject_id}*{file_id}*gait_events*.csv"))
    if fuzzy:
        return fuzzy[0]

    fuzzy2 = list(cond_dir.glob(f"*{subject_id}*{condition}*gait_events*.csv"))
    if fuzzy2:
        return fuzzy2[0]

    return None


def choose_signal_columns(df: pd.DataFrame):
    raw_col = None
    filt_col = None

    raw_candidates = ["heel_fsr_raw", "heel_fsr"]
    filt_candidates = ["heel_fsr_filt", "heel_fsr_filtered", "heel_fsr"]

    for c in raw_candidates:
        if c in df.columns:
            raw_col = c
            break

    for c in filt_candidates:
        if c in df.columns:
            filt_col = c
            break

    if raw_col is None:
        raise ValueError("raw_aligned 中未找到 heel_fsr_raw 或 heel_fsr 列")
    if filt_col is None:
        raise ValueError("raw_aligned 中未找到 heel_fsr_filt / heel_fsr_filtered / heel_fsr 列")

    return raw_col, filt_col


def get_steady_subset(df: pd.DataFrame) -> pd.DataFrame:
    if "activity_label" in df.columns:
        df = df[df["activity_label"] == "walking"].copy()

    if "steady_mask" in df.columns:
        steady = df[df["steady_mask"] == 1].copy()
        if not steady.empty:
            return steady

    return df.copy()


def pick_window_centers(ts_min: float, ts_max: float, window_sec: float):
    usable_start = ts_min + window_sec / 2
    usable_end = ts_max - window_sec / 2

    if usable_end <= usable_start:
        center = (ts_min + ts_max) / 2
        return [center]

    span = usable_end - usable_start
    centers = [
        usable_start + 0.15 * span,
        usable_start + 0.50 * span,
        usable_start + 0.85 * span,
    ]
    return centers


def plot_one_window(df_win: pd.DataFrame, hs_times_win: np.ndarray, raw_col: str, filt_col: str,
                    subject_id: str, condition: str, file_id: str, tag: str, out_path: Path):
    plt.figure(figsize=(14, 4.8))
    plt.plot(df_win["timestamp"], df_win[raw_col], label=raw_col, linewidth=1.2)
    plt.plot(df_win["timestamp"], df_win[filt_col], label=filt_col, linewidth=2.0)

    if len(hs_times_win) > 0:
        y_hs = np.interp(hs_times_win, df_win["timestamp"].to_numpy(), df_win[filt_col].to_numpy())
        plt.scatter(hs_times_win, y_hs, color="red", s=50, label="Detected HS", zorder=5)
        for t in hs_times_win:
            plt.axvline(t, color="red", alpha=0.15, linewidth=1)

    plt.title(f"Local HS QA: {subject_id} {condition} {file_id} | {tag}")
    plt.xlabel("Time (s)")
    plt.ylabel("Heel FSR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def load_targets(project_root: Path):
    processed_qc_dir = project_root / "data" / "processed" / "qc"
    suspicious_csv = processed_qc_dir / "gait_event_summary_suspicious.csv"

    if USE_SUSPICIOUS_ONLY and suspicious_csv.exists():
        df = pd.read_csv(suspicious_csv)
        return df[["subject_id", "file_id", "condition"]].drop_duplicates().to_dict("records")

    return [{"subject_id": s, "file_id": "", "condition": c} for s, c in MANUAL_TARGETS]


def main():
    project_root = Path(__file__).resolve().parents[2]

    data_dir = project_root / "data"
    raw_aligned_dir = data_dir / "raw_aligned"
    gait_events_dir = data_dir / "gait_events"
    out_dir = data_dir / "processed" / "qc" / "local_heelstrike"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = load_targets(project_root)
    if not targets:
        raise ValueError("没有找到目标文件。请检查 suspicious csv 或 MANUAL_TARGETS。")

    for row in targets:
        subject_id = str(row["subject_id"])
        file_id = str(row["file_id"]) if pd.notna(row["file_id"]) else ""
        condition = str(row["condition"])

        raw_path = find_raw_aligned_file(raw_aligned_dir, subject_id, file_id, condition)
        if raw_path is None:
            print(f"[WARN] 找不到 raw_aligned: {subject_id} | {file_id} | {condition}")
            continue

        ev_path = find_gait_event_file(gait_events_dir, subject_id, file_id, condition)
        if ev_path is None:
            print(f"[WARN] 找不到 gait_events: {subject_id} | {file_id} | {condition}")
            continue

        raw_df = pd.read_csv(raw_path)
        ev_df = pd.read_csv(ev_path)

        if raw_df.empty or ev_df.empty:
            print(f"[WARN] 空文件: {subject_id} | {file_id} | {condition}")
            continue

        if "timestamp" not in raw_df.columns or "timestamp" not in ev_df.columns:
            print(f"[WARN] 缺 timestamp: {subject_id} | {file_id} | {condition}")
            continue

        raw_col, filt_col = choose_signal_columns(raw_df)
        steady_df = get_steady_subset(raw_df)

        if steady_df.empty:
            print(f"[WARN] steady / walking 数据为空: {subject_id} | {file_id} | {condition}")
            continue

        ts_min = float(steady_df["timestamp"].min())
        ts_max = float(steady_df["timestamp"].max())
        centers = pick_window_centers(ts_min, ts_max, WINDOW_SEC)

        hs_times = ev_df["timestamp"].astype(float).to_numpy()
        hs_times = hs_times[(hs_times >= ts_min) & (hs_times <= ts_max)]

        tags = ["start", "mid", "end"] if len(centers) == 3 else ["mid"]

        for center, tag in zip(centers, tags):
            start_t = center - WINDOW_SEC / 2
            end_t = center + WINDOW_SEC / 2

            df_win = steady_df[(steady_df["timestamp"] >= start_t) & (steady_df["timestamp"] <= end_t)].copy()
            hs_win = hs_times[(hs_times >= start_t) & (hs_times <= end_t)]

            if df_win.empty:
                continue

            safe_file_id = file_id if file_id else raw_path.stem
            out_path = out_dir / f"{subject_id}_{condition}_{safe_file_id}_{tag}_local_hs.png"

            plot_one_window(
                df_win=df_win,
                hs_times_win=hs_win,
                raw_col=raw_col,
                filt_col=filt_col,
                subject_id=subject_id,
                condition=condition,
                file_id=safe_file_id,
                tag=tag,
                out_path=out_path,
            )

            print(f"[OK] saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()