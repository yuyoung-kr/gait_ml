from pathlib import Path
import numpy as np
import pandas as pd


def compute_cycle_stats(hs_times: np.ndarray) -> dict:
    if len(hs_times) < 2:
        return {
            "n_cycles": 0,
            "median_cycle_duration_s": np.nan,
            "mean_cycle_duration_s": np.nan,
            "std_cycle_duration_s": np.nan,
            "cv_cycle_duration": np.nan,
            "min_cycle_duration_s": np.nan,
            "max_cycle_duration_s": np.nan,
            "n_short_cycles": 0,
            "n_long_cycles": 0,
            "outlier_ratio": np.nan,
        }

    cycle_durations = np.diff(hs_times)
    median_cd = float(np.median(cycle_durations))
    mean_cd = float(np.mean(cycle_durations))
    std_cd = float(np.std(cycle_durations, ddof=0))
    cv_cd = float(std_cd / mean_cd) if mean_cd > 0 else np.nan

    short_mask = cycle_durations < (0.5 * median_cd)
    long_mask = cycle_durations > (1.8 * median_cd)

    n_short = int(short_mask.sum())
    n_long = int(long_mask.sum())
    n_cycles = int(len(cycle_durations))
    outlier_ratio = float((n_short + n_long) / n_cycles) if n_cycles > 0 else np.nan

    return {
        "n_cycles": n_cycles,
        "median_cycle_duration_s": median_cd,
        "mean_cycle_duration_s": mean_cd,
        "std_cycle_duration_s": std_cd,
        "cv_cycle_duration": cv_cd,
        "min_cycle_duration_s": float(np.min(cycle_durations)),
        "max_cycle_duration_s": float(np.max(cycle_durations)),
        "n_short_cycles": n_short,
        "n_long_cycles": n_long,
        "outlier_ratio": outlier_ratio,
    }


def compute_steady_duration(raw_df: pd.DataFrame) -> float:
    needed = {"activity_label", "steady_mask", "timestamp"}
    if not needed.issubset(raw_df.columns):
        return np.nan

    mask = (raw_df["activity_label"] == "walking") & (raw_df["steady_mask"] == 1)
    ts = raw_df.loc[mask, "timestamp"].to_numpy()

    if len(ts) < 2:
        return np.nan

    duration = float(ts[-1] - ts[0])
    return duration if duration > 0 else np.nan


def find_gait_event_file(gait_events_dir: Path, condition: str, subject_id: str, file_id: str) -> Path | None:
    cond_dir = gait_events_dir / condition
    if not cond_dir.exists():
        return None

    exact = cond_dir / f"{subject_id}_{file_id}_{condition}_gait_events.csv"
    if exact.exists():
        return exact

    candidates = list(cond_dir.glob(f"*{file_id}*gait_events*.csv"))
    if candidates:
        return candidates[0]

    return None


def main():
    project_root = Path(__file__).resolve().parents[2]

    data_dir = project_root / "data"
    raw_aligned_dir = data_dir / "raw_aligned"
    gait_events_dir = data_dir / "gait_events"
    cycles_dir = data_dir / "cycles"
    out_dir = data_dir / "processed" / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    raw_files = sorted(raw_aligned_dir.glob("*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"未找到 raw_aligned 文件: {raw_aligned_dir}")

    valid_conditions = {"walking_2kmh", "walking_4kmh", "walking_6kmh"}

    for raw_path in raw_files:
        raw_df = pd.read_csv(raw_path)
        if raw_df.empty:
            continue

        condition = str(raw_df["condition"].iloc[0])
        if condition not in valid_conditions:
            continue

        subject_id = str(raw_df["subject_id"].iloc[0])
        file_id = str(raw_df["file_id"].iloc[0])
        trial_id = str(raw_df["trial_id"].iloc[0]) if "trial_id" in raw_df.columns else ""

        duration_s = compute_steady_duration(raw_df)

        gait_event_file = find_gait_event_file(gait_events_dir, condition, subject_id, file_id)
        if gait_event_file is not None and gait_event_file.exists():
            ev_df = pd.read_csv(gait_event_file)
        else:
            ev_df = pd.DataFrame(columns=["timestamp", "event_type", "event_idx", "event_valid"])

        if not ev_df.empty and "timestamp" in ev_df.columns:
            hs_times = np.sort(ev_df["timestamp"].astype(float).to_numpy())
        else:
            hs_times = np.array([], dtype=float)

        n_hs = int(len(hs_times))
        hs_rate_hz = float(n_hs / duration_s) if pd.notna(duration_s) and duration_s > 0 else np.nan

        cycle_stats = compute_cycle_stats(hs_times)

        cycle_pattern = f"{subject_id}_{condition}_cycle_*.csv"
        cycle_files = list((cycles_dir / condition).glob(cycle_pattern))
        n_cycle_files = len(cycle_files)
        n_cycles_expected = max(n_hs - 1, 0)
        cycles_match_expected = int(n_cycle_files == n_cycles_expected)

        suspicious = 0
        notes = []

        if n_hs < 5:
            suspicious = 1
            notes.append("too_few_hs")

        if n_cycles_expected != n_cycle_files:
            suspicious = 1
            notes.append("cycle_count_mismatch")

        if pd.notna(cycle_stats["outlier_ratio"]) and cycle_stats["outlier_ratio"] > 0.15:
            suspicious = 1
            notes.append("high_outlier_ratio")

        rows.append({
            "subject_id": subject_id,
            "file_id": file_id,
            "trial_id": trial_id,
            "condition": condition,
            "duration_s": duration_s,
            "n_hs": n_hs,
            "n_cycles": cycle_stats["n_cycles"],
            "n_cycles_expected": n_cycles_expected,
            "n_cycle_files": n_cycle_files,
            "cycles_match_expected": cycles_match_expected,
            "hs_rate_hz": hs_rate_hz,
            "median_cycle_duration_s": cycle_stats["median_cycle_duration_s"],
            "mean_cycle_duration_s": cycle_stats["mean_cycle_duration_s"],
            "std_cycle_duration_s": cycle_stats["std_cycle_duration_s"],
            "cv_cycle_duration": cycle_stats["cv_cycle_duration"],
            "min_cycle_duration_s": cycle_stats["min_cycle_duration_s"],
            "max_cycle_duration_s": cycle_stats["max_cycle_duration_s"],
            "n_short_cycles": cycle_stats["n_short_cycles"],
            "n_long_cycles": cycle_stats["n_long_cycles"],
            "outlier_ratio": cycle_stats["outlier_ratio"],
            "suspicious": suspicious,
            "notes": ";".join(notes) if notes else "",
        })

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        raise ValueError("没有生成任何 gait_event_summary 行，请检查 raw_aligned / gait_events / cycles。")

    summary_df = summary_df.sort_values(["subject_id", "condition", "trial_id"]).reset_index(drop=True)

    out_csv = out_dir / "gait_event_summary.csv"
    summary_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"已生成: {out_csv}")

    suspicious_df = summary_df.loc[summary_df["suspicious"] == 1].copy()
    if not suspicious_df.empty:
        suspicious_csv = out_dir / "gait_event_summary_suspicious.csv"
        suspicious_df.to_csv(suspicious_csv, index=False, encoding="utf-8-sig")
        print(f"已生成: {suspicious_csv}")

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
    