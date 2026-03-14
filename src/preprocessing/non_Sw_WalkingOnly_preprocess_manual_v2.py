# ============================== preprocess_gait_v8_final.py ==============================
# 状态：Walking-Only Phase Preprocessing Stable Version
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import medfilt
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

# ============================ 配置 ============================
RAW_DIR = Path("data/raw/yy260311_walking_only_no_switch")
OUTPUT_BASE = Path("data")
METADATA_FILE = Path("metadata.csv")
TARGET_FS = 200.0
WINDOW_MS_LIST = [150]
STRIDE_SAMPLES = 10          # ← 滑窗步长（默认10点=50ms）
HEEL_THRESHOLD = 350
MIN_CYCLE_DURATION_S = 0.4
FSR_KERNEL = 7
VALID_CONDITIONS = ['standing', 'walking_2kmh', 'walking_4kmh', 'walking_6kmh']
TAIL_TRIM_S = 3.0
def load_metadata():
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"请创建 {METADATA_FILE}")

    meta = pd.read_csv(METADATA_FILE)
    meta['filename'] = meta['filename'].str.lower().str.replace('.csv', '', regex=False)
    meta['condition'] = meta['condition'].str.lower().str.strip()

    if meta['filename'].duplicated().any():
        raise ValueError("metadata.csv 中 filename 重复！")

    required_cols = ['subject_id', 'condition', 'trial_id', 'walking_start_s', 'walking_end_s']

    for col in ['walking_start_s', 'walking_end_s']:
        if col not in meta.columns:
            meta[col] = np.nan

    base_required = ['subject_id', 'condition', 'trial_id']
    if meta[base_required].isnull().any().any():
        raise ValueError("metadata.csv 中存在空字段！")

    walk_rows = meta['condition'].isin(['walking_2kmh', 'walking_4kmh', 'walking_6kmh'])
    if meta.loc[walk_rows, ['walking_start_s', 'walking_end_s']].isnull().any().any():
        raise ValueError("walking 文件必须填写 walking_start_s / walking_end_s")

    meta['walking_start_s'] = pd.to_numeric(meta['walking_start_s'], errors='coerce')
    meta['walking_end_s'] = pd.to_numeric(meta['walking_end_s'], errors='coerce')

    if not meta['condition'].isin(VALID_CONDITIONS).all():
        raise ValueError(f"condition 必须是 {VALID_CONDITIONS}")

    meta['activity_label'] = np.where(meta['condition'].eq('standing'), 'standing', 'walking')

    if meta[['subject_id', 'condition', 'trial_id']].duplicated().any():
        raise ValueError("subject_id + condition + trial_id 组合重复！")

    return meta.set_index('filename')


METADATA = load_metadata()


def get_metadata(filename: str):
    stem = filename.lower().replace(".csv", "")
    if stem not in METADATA.index:
        raise ValueError(f"❌ 文件名未在 metadata.csv 中: {filename}")
    row = METADATA.loc[stem]
    return (
    row['subject_id'],
    row['condition'],
    row['trial_id'],
    row['activity_label'],
    row['walking_start_s'],
    row['walking_end_s']
    )

# ============================ 异常检查 + 修复 + 复检 ============================
def check_and_fix_anomalies(df, subject_id, condition, trial_id):
    df = df.copy()

    if df['timestamp'].isna().any():
        warnings.warn(f"[{subject_id} {condition} {trial_id}] NaN 时间戳 → 已填充")
        df['timestamp'] = df['timestamp'].ffill().bfill()

    if (df['timestamp'].diff() < 0).any():
        warnings.warn(f"[{subject_id} {condition} {trial_id}] 时间倒退 → 已排序")
        df = df.sort_values('timestamp').reset_index(drop=True)

    if df['timestamp'].duplicated().any():
        warnings.warn(f"[{subject_id} {condition} {trial_id}] 重复时间戳 → 已去重")
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

    sensor_cols = [
        'knee_angle', 'knee_ang_vel',
        'ax', 'ay', 'az', 'gx', 'gy', 'gz',
        'toe_fsr', 'heel_fsr'
    ]

    for col in sensor_cols:
        if df[col].isna().any():
            df[col] = df[col].interpolate().bfill().ffill()

    for col in sensor_cols + ['timestamp']:
        if df[col].isna().any():
            raise ValueError(f"❌ {col} 修复后仍有 NaN，跳过文件 {subject_id}_{condition}")

    print(f"✅ 异常检查+修复+复检完成: {subject_id} {condition} {trial_id}")
    return df

def load_raw(file_path: Path):
    df = pd.read_csv(file_path, header=None)

    if df.shape[1] == 14:
        cols = [
            'knee_angle', 'knee_ang_vel', 'unused_angle',
            'ax', 'ay', 'az', 'gx', 'gy', 'gz',
            'toe_fsr', 'heel_fsr',
            'timestamp', 'event_code', 'imu_sample_idx'
        ]
    elif df.shape[1] == 13:
        cols = [
            'knee_angle', 'knee_ang_vel', 'unused_angle',
            'ax', 'ay', 'az', 'gx', 'gy', 'gz',
            'toe_fsr', 'heel_fsr',
            'timestamp', 'event_code'
        ]
    else:
        raise ValueError(f"❌ 文件列数异常: {df.shape[1]} 列（只支持13或14列）")

    df.columns = cols

    if 'imu_sample_idx' not in df.columns:
        df['imu_sample_idx'] = np.nan

    df['timestamp'] = df['timestamp'].astype(float)
    df['event_code'] = df['event_code'].fillna(0).astype(int)

    return df

def build_imu_update_flag(df):
    df = df.copy()

    if df['imu_sample_idx'].isna().all():
        imu_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
        changed = df[imu_cols].diff().abs().max(axis=1).fillna(1) > 1e-6
        df['imu_sample_idx'] = changed.cumsum()

    df['imu_sample_idx'] = df['imu_sample_idx'].ffill().bfill().astype(int)
    df['imu_update_flag'] = df['imu_sample_idx'].ne(df['imu_sample_idx'].shift()).astype(int)
    df.loc[df.index[0], 'imu_update_flag'] = 1
    return df


def extract_event_table(df):
    ev = df[['timestamp', 'event_code']].copy()
    ev['event_code'] = ev['event_code'].fillna(0).astype(int)

    pulse_start = ev['event_code'].ne(0) & ev['event_code'].shift(fill_value=0).eq(0)
    ev = ev.loc[pulse_start].reset_index(drop=True)
    return ev


def map_events_to_grid(t_grid, event_table):
    event_code = np.zeros(len(t_grid), dtype=int)

    if event_table.empty:
        return event_code

    for _, row in event_table.iterrows():
        ts = float(row['timestamp'])
        code = int(row['event_code'])

        idx = np.searchsorted(t_grid, ts)
        if idx <= 0:
            nearest = 0
        elif idx >= len(t_grid):
            nearest = len(t_grid) - 1
        else:
            left = idx - 1
            right = idx
            nearest = right if abs(t_grid[right] - ts) <= abs(ts - t_grid[left]) else left

        event_code[nearest] = code

    return event_code


def annotate_file_labels(df, condition, walking_start_s=None, walking_end_s=None):
    df = df.copy()
    df['condition'] = condition
    df['activity_label'] = 'standing'
    df['steady_mask'] = 0

    t0 = float(df['timestamp'].iloc[0])
    t_rel = df['timestamp'] - t0

    if condition == 'standing':
        df['activity_label'] = 'standing'
        df['steady_mask'] = 1
        return df

    if pd.isna(walking_start_s) or pd.isna(walking_end_s):
        raise ValueError(f"{condition} 文件必须提供 walking_start_s 和 walking_end_s")

    walking_start_s = float(walking_start_s)
    walking_end_s = float(walking_end_s)

    walk_mask = (t_rel >= walking_start_s) & (t_rel < walking_end_s)

    df.loc[walk_mask, 'activity_label'] = 'walking'
    df.loc[walk_mask, 'steady_mask'] = 1

    trim_end_s = walking_end_s - TAIL_TRIM_S
    if trim_end_s > walking_start_s:
        df.loc[(t_rel >= trim_end_s) & (t_rel < walking_end_s), 'steady_mask'] = 0

    return df

def align_and_resample(df):
    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = build_imu_update_flag(df)

    event_table = extract_event_table(df)

    t_start = df['timestamp'].min()
    t_end = df['timestamp'].max()
    new_index = np.arange(t_start, t_end + 1e-9, 1.0 / TARGET_FS)

    df_res = pd.DataFrame({'timestamp': new_index})

    kin_cols = ['knee_angle', 'knee_ang_vel', 'toe_fsr', 'heel_fsr']
    for col in kin_cols:
        df_res[col] = np.interp(new_index, df['timestamp'], df[col])

    imu_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    imu_df = df.loc[df['imu_update_flag'] == 1, ['timestamp', 'imu_sample_idx'] + imu_cols].copy()
    imu_df = imu_df.drop_duplicates(subset=['imu_sample_idx']).reset_index(drop=True)

    for col in imu_cols:
        df_res[col] = np.interp(new_index, imu_df['timestamp'], imu_df[col])

    df_res = pd.merge_asof(
        df_res,
        imu_df[['timestamp', 'imu_sample_idx']],
        on='timestamp',
        direction='backward'
    )

    df_res['imu_sample_idx'] = df_res['imu_sample_idx'].ffill().bfill().astype(int)
    df_res['imu_update_flag'] = df_res['imu_sample_idx'].ne(df_res['imu_sample_idx'].shift()).astype(int)
    df_res.loc[df_res.index[0], 'imu_update_flag'] = 1

    df_res['event_code'] = map_events_to_grid(new_index, event_table)

    return df_res

def detect_heel_strikes(df_walk):
    fsr_filt = medfilt(df_walk['heel_fsr'].to_numpy(), FSR_KERNEL)

    above = fsr_filt >= HEEL_THRESHOLD
    rising_edges = np.where((~above[:-1]) & (above[1:]))[0] + 1

    min_gap = int(MIN_CYCLE_DURATION_S * TARGET_FS)
    hs_idx = []
    last_idx = -10**9

    for idx in rising_edges:
        if idx - last_idx >= min_gap:
            hs_idx.append(idx)
            last_idx = idx

    return np.array(hs_idx, dtype=int), fsr_filt

def save_heelstrike_qa(df_walk, hs_idx, fsr_filt, subject_id, condition, trial_id):
    plt.figure(figsize=(12, 4))
    plt.plot(df_walk['timestamp'], df_walk['heel_fsr'], label='heel_fsr_raw')
    plt.plot(df_walk['timestamp'], fsr_filt, label='heel_fsr_filt')
    if len(hs_idx) > 0:
        plt.scatter(
            df_walk.iloc[hs_idx]['timestamp'],
            df_walk.iloc[hs_idx]['heel_fsr'],
            color='red',
            s=30,
            label='Detected HS'
        )
    plt.title(f'Heel-Strike QA: {subject_id} {condition} {trial_id} | {len(hs_idx)} strikes')
    plt.legend()

    out_dir = OUTPUT_BASE / "qa_heelstrike"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"{subject_id}_{condition}_{trial_id}_qa.png", dpi=150, bbox_inches='tight')
    plt.close()

def process_one_raw(raw_path: Path):
    subject_id, condition, trial_id, _, walking_start_s, walking_end_s = get_metadata(raw_path.name)
    file_id = raw_path.stem.lower()

    print(f"处理 {raw_path.name} → {subject_id} | {condition} | {trial_id}")

    df = load_raw(raw_path)
    df = check_and_fix_anomalies(df, subject_id, condition, trial_id)
    df_aligned = align_and_resample(df)
    
    df_aligned = annotate_file_labels(
    df_aligned,
    condition,
    walking_start_s=walking_start_s,
    walking_end_s=walking_end_s
    )

    df_aligned['subject_id'] = subject_id
    df_aligned['file_id'] = file_id
    df_aligned['trial_id'] = trial_id

    df_aligned['cycle_id'] = 0
    df_aligned['cycle_duration_s'] = np.nan
    df_aligned['phase_pct'] = np.nan
    df_aligned['phase_valid_mask'] = 0

    gait_events_dir = OUTPUT_BASE / "gait_events" / condition
    gait_events_dir.mkdir(parents=True, exist_ok=True)

    cycles_dir = OUTPUT_BASE / "cycles" / condition
    cycles_dir.mkdir(parents=True, exist_ok=True)

    phase_temp_dir = OUTPUT_BASE / "phase_summary_temp"
    phase_temp_dir.mkdir(parents=True, exist_ok=True)

    if condition != 'standing':
        walk_mask = (df_aligned['activity_label'] == 'walking') & (df_aligned['steady_mask'] == 1)
        df_walk = df_aligned.loc[walk_mask].copy()
        walk_global_idx = df_walk.index.to_numpy()

        if not df_walk.empty:
            hs_local_idx, fsr_filt = detect_heel_strikes(df_walk)
            save_heelstrike_qa(df_walk, hs_local_idx, fsr_filt, subject_id, condition, trial_id)

            gait_events_df = pd.DataFrame({
                'subject_id': subject_id,
                'file_id': file_id,
                'trial_id': trial_id,
                'condition': condition,
                'timestamp': df_walk.iloc[hs_local_idx]['timestamp'].to_numpy() if len(hs_local_idx) > 0 else [],
                'event_type': ['heel_strike'] * len(hs_local_idx),
                'event_idx': np.arange(1, len(hs_local_idx) + 1),
                'event_valid': [1] * len(hs_local_idx)
            })
            gait_events_df.to_csv(
                gait_events_dir / f"{subject_id}_{file_id}_{condition}_gait_events.csv",
                index=False
            )

            for i in range(len(hs_local_idx) - 1):
                start_local = hs_local_idx[i]
                end_local = hs_local_idx[i + 1]

                start_global = walk_global_idx[start_local]
                end_global = walk_global_idx[end_local]

                duration = df_aligned.loc[end_global, 'timestamp'] - df_aligned.loc[start_global, 'timestamp']
                if duration < MIN_CYCLE_DURATION_S:
                    continue

                cycle_rows = np.arange(start_global, end_global)
                if len(cycle_rows) == 0:
                    continue

                phase_values = np.arange(len(cycle_rows), dtype=float) / float(len(cycle_rows))

                df_aligned.loc[cycle_rows, 'cycle_id'] = i + 1
                df_aligned.loc[cycle_rows, 'cycle_duration_s'] = duration
                df_aligned.loc[cycle_rows, 'phase_pct'] = phase_values
                df_aligned.loc[cycle_rows, 'phase_valid_mask'] = 1

                cycle_df = df_aligned.loc[cycle_rows, [
                    'subject_id', 'file_id', 'trial_id',
                    'activity_label', 'condition',
                    'timestamp',
                    'knee_angle', 'knee_ang_vel',
                    'ax', 'ay', 'az', 'gx', 'gy', 'gz',
                    'heel_fsr', 'toe_fsr'
                ]].copy()

                cycle_df['cycle_id'] = i + 1
                cycle_df['cycle_start_time'] = float(df_aligned.loc[start_global, 'timestamp'])
                cycle_df['cycle_end_time'] = float(df_aligned.loc[end_global, 'timestamp'])

                cycle_path = cycles_dir / f"{subject_id}_{file_id}_{condition}_cycle_{i+1:03d}.csv"
                cycle_df.to_csv(cycle_path, index=False)

    raw_aligned_dir = OUTPUT_BASE / "raw_aligned"
    raw_aligned_dir.mkdir(parents=True, exist_ok=True)

    raw_aligned_cols = [
        'subject_id', 'file_id', 'trial_id',
        'activity_label', 'condition',
        'timestamp',
        'knee_angle', 'knee_ang_vel',
        'ax', 'ay', 'az', 'gx', 'gy', 'gz',
        'heel_fsr', 'toe_fsr',
        'event_code',
        'imu_sample_idx', 'imu_update_flag',
        'steady_mask',
        'cycle_id', 'cycle_duration_s',
        'phase_pct', 'phase_valid_mask'
    ]
    df_aligned[raw_aligned_cols].to_csv(
        raw_aligned_dir / f"{subject_id}_{file_id}_{condition}_{trial_id}_aligned.csv",
        index=False
    )

    if condition != 'standing':
        phase_df = df_aligned.loc[df_aligned['phase_valid_mask'] == 1, [
            'subject_id', 'file_id', 'trial_id',
            'activity_label', 'condition',
            'cycle_id', 'timestamp', 'phase_pct', 'cycle_duration_s',
            'knee_angle', 'knee_ang_vel',
            'ax', 'ay', 'az', 'gx', 'gy', 'gz',
            'heel_fsr', 'toe_fsr',
            'phase_valid_mask'
        ]].copy()

        phase_df.to_csv(
            phase_temp_dir / f"{subject_id}_{file_id}_{condition}_{trial_id}_phase_summary.csv",
            index=False
        )

    feat_cols = ['knee_angle', 'knee_ang_vel', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']

    for w_ms in WINDOW_MS_LIST:
        seg_dir = OUTPUT_BASE / "segments" / f"window_{w_ms}ms"
        seg_dir.mkdir(parents=True, exist_ok=True)

        window_samples = int(w_ms * TARGET_FS / 1000)

        segments_list = []
        labels_list = []

        for i in range(0, len(df_aligned) - window_samples + 1, STRIDE_SAMPLES):
            win = df_aligned.iloc[i:i + window_samples]
            end_row = win.iloc[-1]

            if not (win['steady_mask'] == 1).all():
                continue

            phase_valid = int(end_row['phase_valid_mask']) == 1
            phase_pct = float(end_row['phase_pct']) if phase_valid else np.nan

            regression_mask = int(
                (end_row['activity_label'] == 'walking') and phase_valid
            )

            segments_list.append(win[feat_cols].to_numpy(dtype=np.float32))

            labels_list.append({
                'segment_id': len(labels_list),
                'subject_id': subject_id,
                'file_id': file_id,
                'trial_id': trial_id,
                'activity_label': end_row['activity_label'],
                'condition': end_row['condition'],
                'segment_start_time': float(win['timestamp'].iloc[0]),
                'segment_end_time': float(end_row['timestamp']),
                'window_ms': w_ms,
                'phase_pct': phase_pct,
                'regression_mask': regression_mask,
                'steady_mask': int(end_row['steady_mask'])
            })

        if segments_list:
            seg_array = np.stack(segments_list)
            np.save(seg_dir / f"{subject_id}_{file_id}_{condition}_{trial_id}_segments.npy", seg_array)
            pd.DataFrame(labels_list).to_csv(
                seg_dir / f"{subject_id}_{file_id}_{condition}_{trial_id}_segment_labels.csv",
                index=False
            )

    print(f"✅ {subject_id} {condition} {trial_id} 完成")

def merge_phase_summary_by_condition():
    temp_dir = OUTPUT_BASE / "phase_summary_temp"
    out_dir = OUTPUT_BASE / "phase_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    for condition in ['walking_2kmh', 'walking_4kmh', 'walking_6kmh']:
        files = sorted(temp_dir.glob(f"*_{condition}_*_phase_summary.csv"))
        if not files:
            continue

        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df.to_csv(out_dir / f"phase_summary_{condition}.csv", index=False)
        print(f"✅ 已汇总: phase_summary_{condition}.csv")

if __name__ == "__main__":
    for f in sorted(RAW_DIR.glob("*.csv")):
        process_one_raw(f)

    merge_phase_summary_by_condition()
    print("\n🎉 Raw preprocessing 完成！")