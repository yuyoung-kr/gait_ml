# ============================== preprocess_gait_v8_final.py ==============================
# 状态：Walking-Only Phase Preprocessing Stable Version（已处理全部11个优化点，可稳定跑通）
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, medfilt
from pathlib import Path
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

# ============================ 配置 ============================
RAW_DIR = Path("data/raw")
OUTPUT_BASE = Path("data")
METADATA_FILE = Path("metadata.csv")
TARGET_FS = 200.0
WINDOW_MS_LIST = [150, 600]
STRIDE_SAMPLES = 10          # ← 滑窗步长（默认10点=50ms）
HEEL_THRESHOLD = 350
MIN_CYCLE_DURATION_S = 0.4
FSR_KERNEL = 7

def load_metadata():
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"请创建 {METADATA_FILE}")
    meta = pd.read_csv(METADATA_FILE)
    meta['filename'] = meta['filename'].str.lower().str.replace('.csv', '')
    # 完整校验
    if meta['filename'].duplicated().any():
        raise ValueError("metadata.csv 中 filename 重复！")
    if meta[['subject_id', 'speed', 'trial_id']].isnull().any().any():
        raise ValueError("metadata.csv 中存在空字段！")
    if not meta['speed'].isin(['1kmh', '4kmh', '6kmh']).all():
        raise ValueError("speed 必须是 1kmh/4kmh/6kmh 之一！")
    if meta[['subject_id', 'speed', 'trial_id']].duplicated().any():
        raise ValueError("subject_id + speed + trial_id 组合重复！")
    return meta.set_index('filename')

METADATA = load_metadata()

def get_metadata(filename: str):
    stem = filename.lower().replace(".csv", "")
    if stem not in METADATA.index:
        raise ValueError(f"❌ 文件名未在 metadata.csv 中: {filename}")
    row = METADATA.loc[stem]
    return row['subject_id'], row['speed'], row['trial_id']

# ============================ 异常检查 + 修复 + 复检 ============================
def check_and_fix_anomalies(df, subject_id, speed, trial_id):
    # timestamp
    if df['timestamp'].isna().any():
        warnings.warn(f"[{subject_id} {speed} {trial_id}] NaN 时间戳 → 已填充")
        df['timestamp'] = df['timestamp'].ffill().bfill()
    if (df['timestamp'].diff() < 0).any():
        warnings.warn(f"[{subject_id} {speed} {trial_id}] 时间倒退 → 已排序")
        df = df.sort_values('timestamp').reset_index(drop=True)
    if df['timestamp'].duplicated().any():
        warnings.warn(f"[{subject_id} {speed} {trial_id}] 重复时间戳 → 已去重")
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    # 传感器全覆盖
    sensor_cols = ['knee_angle','knee_ang_vel','imu_angle','ax','ay','az','gx','gy','gz','fsr_toe','fsr_heel']
    for col in sensor_cols:
        if df[col].isna().any():
            df[col] = df[col].interpolate().bfill().ffill()
    # 复检
    for col in sensor_cols + ['timestamp']:
        if df[col].isna().any():
            raise ValueError(f"❌ {col} 修复后仍有 NaN，跳过文件 {subject_id}_{speed}")
    print(f"✅ 异常检查+修复+复检完成: {subject_id} {speed} {trial_id}")
    return df

def load_raw(file_path: Path):
    df = pd.read_csv(file_path, header=None)
    if df.shape[1] == 14:
        cols = ['knee_angle','knee_ang_vel','imu_angle','ax','ay','az','gx','gy','gz',
                'fsr_toe','fsr_heel','timestamp','event_code','imu_sample_idx']
    elif df.shape[1] == 13:
        cols = ['knee_angle','knee_ang_vel','imu_angle','ax','ay','az','gx','gy','gz',
                'fsr_toe','fsr_heel','timestamp','event_code']
    else:
        raise ValueError(f"❌ 文件列数异常: {df.shape[1]} 列（只支持13或14列）")
    df.columns = cols
    if 'imu_sample_idx' not in df.columns:
        df['imu_sample_idx'] = np.nan
    df['timestamp'] = df['timestamp'].astype(float)
    return df

def align_and_resample(df):
    df = df.copy()
    if df['imu_sample_idx'].isna().all():
        imu_cols = ['ax','ay','az','gx','gy','gz']
        df['imu_sample_idx'] = (df[imu_cols].diff().abs().max(axis=1) > 0.001).cumsum()
    t_start, t_end = df['timestamp'].min(), df['timestamp'].max()
    new_index = np.arange(t_start, t_end + 1e-6, 1.0 / TARGET_FS)
    cont_cols = ['knee_angle','knee_ang_vel','imu_angle','ax','ay','az','gx','gy','gz','fsr_toe','fsr_heel']
    df_res = pd.DataFrame({'timestamp': new_index})
    for col in cont_cols:
        df_res[col] = np.interp(new_index, df['timestamp'], df[col])
    disc_cols = ['event_code','imu_sample_idx']
    df_disc = df[['timestamp'] + disc_cols].copy()
    df_res = pd.merge_asof(df_res, df_disc, on='timestamp', direction='nearest')
    return df_res

def detect_heel_strikes(df):
    fsr = medfilt(df['fsr_heel'].values, FSR_KERNEL)
    peaks_down = find_peaks(-fsr, distance=int(MIN_CYCLE_DURATION_S*TARGET_FS), prominence=80, height=-HEEL_THRESHOLD)[0]
    peaks_up = find_peaks(fsr, distance=int(MIN_CYCLE_DURATION_S*TARGET_FS), prominence=80, height=HEEL_THRESHOLD)[0]
    return peaks_down if len(peaks_down) >= len(peaks_up) else peaks_up

def save_heelstrike_qa(df, hs_idx, subject_id, speed, trial_id):
    plt.figure(figsize=(12, 4))
    plt.plot(df['timestamp'], df['fsr_heel'], label='FSR Heel')
    plt.plot(df['timestamp'], medfilt(df['fsr_heel'], 7), label='medfilt')
    plt.scatter(df.iloc[hs_idx]['timestamp'], df.iloc[hs_idx]['fsr_heel'], color='red', s=30, label='Detected HS')
    plt.title(f'Heel-Strike QA: {subject_id} {speed} {trial_id} | {len(hs_idx)} strikes')
    plt.legend()
    (OUTPUT_BASE / "qa_heelstrike").mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_BASE / "qa_heelstrike" / f"{subject_id}_{speed}_{trial_id}_qa.png", dpi=150, bbox_inches='tight')
    plt.close()

def process_one_raw(raw_path: Path):
    subject_id, speed, trial_id = get_metadata(raw_path.name)
    print(f"处理 {raw_path.name} → {subject_id} | {speed} | {trial_id}")

    df = load_raw(raw_path)
    df = check_and_fix_anomalies(df, subject_id, speed, trial_id)
    df_aligned = align_and_resample(df)

    # raw_aligned
    (OUTPUT_BASE / "raw_aligned").mkdir(parents=True, exist_ok=True)
    df_aligned.to_csv(OUTPUT_BASE / "raw_aligned" / f"{subject_id}_{speed}_{trial_id}_aligned.csv", index=False)

    # Cycles
    hs_idx = detect_heel_strikes(df_aligned)
    save_heelstrike_qa(df_aligned, hs_idx, subject_id, speed, trial_id)

    df_aligned['phase_pct'] = 0.0
    df_aligned['cycle_id'] = 0

    cycle_dir = OUTPUT_BASE / "cycles" / f"{subject_id}_walking-level-{speed}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    cycle_info = []

    for i in range(len(hs_idx)-1):
        start, end = hs_idx[i], hs_idx[i+1]
        duration = df_aligned.iloc[end]['timestamp'] - df_aligned.iloc[start]['timestamp']
        if duration < MIN_CYCLE_DURATION_S: continue

        phase_values = np.linspace(0, 100, end - start + 1)
        df_aligned.loc[start:end, 'phase_pct'] = phase_values
        df_aligned.loc[start:end, 'cycle_id'] = i + 1

        cycle_df = df_aligned.iloc[start:end+1].copy()
        cycle_df['start_time'] = cycle_df['timestamp'].iloc[0]
        cycle_df['end_time'] = cycle_df['timestamp'].iloc[-1]
        cycle_df['subject_id'] = subject_id
        cycle_df['activity'] = "walking"
        cycle_df['speed'] = speed
        cycle_path = cycle_dir / f"{subject_id}_walking-level-{speed}_{trial_id}_cycle_{i+1:03d}.csv"
        cycle_df.to_csv(cycle_path, index=False)
        cycle_info.append({'start_idx': start, 'end_idx': end, 'duration': duration, 'cycle_id': i+1})

    # activity_label（walking-only 明确标注）
    df_aligned['activity'] = 'standing'
    walking_active = False
    for idx, row in df_aligned.iterrows():
        if row['event_code'] == 1: walking_active = True
        elif row['event_code'] == 2: walking_active = False
        df_aligned.loc[idx, 'activity'] = 'walking' if walking_active else 'standing'
    df_aligned.loc[df_aligned['cycle_id'] > 0, 'activity'] = 'walking'

    df_aligned['subject_id'] = subject_id
    df_aligned['speed'] = speed

    # cycle_duration（已回填）
    df_aligned['cycle_duration'] = 0.0
    for c in cycle_info:
        df_aligned.loc[c['start_idx']:c['end_idx'], 'cycle_duration'] = c['duration']

    # phase_summary
    summary_path = OUTPUT_BASE / "phase_summary" / f"phase_summary_{subject_id}_walking-level-{speed}_{trial_id}.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_cols = ['subject_id','activity','speed','cycle_id','phase_pct',
                    'knee_angle','knee_ang_vel','ax','ay','az','gx','gy','gz',
                    'fsr_toe','fsr_heel','timestamp']
    summary_df = df_aligned[df_aligned['cycle_id'] > 0][summary_cols]
    summary_df.to_csv(summary_path, index=False)

    # segments（regression_mask 绑定 cycle_id + 删冗余遍历 + 滑窗步长）
    for w_ms in WINDOW_MS_LIST:
        seg_dir = OUTPUT_BASE / "segments" / f"window_{w_ms}ms"
        seg_dir.mkdir(parents=True, exist_ok=True)
        h5_path = seg_dir / f"{subject_id}_{speed}_{trial_id}_segments.h5"
        
        window_samples = int(w_ms * TARGET_FS / 1000)
        feat_cols = ['knee_angle','knee_ang_vel','ax','ay','az','gx','gy','gz']
        
        segments_list, meta_list = [], []
        
        for i in range(0, len(df_aligned) - window_samples + 1, STRIDE_SAMPLES):
            win = df_aligned.iloc[i:i+window_samples]
            end_row = win.iloc[-1]
            
            valid_phase = int(end_row['cycle_id'] > 0)
            reg_mask = valid_phase
            cycle_dur = end_row['cycle_duration']   # ← 直接使用列（删掉 for c 循环）
            
            if valid_phase:
                phi = end_row['phase_pct'] / 100.0
                x = np.cos(2 * np.pi * phi)
                y = np.sin(2 * np.pi * phi)
                r = 100.0 / cycle_dur if cycle_dur > 0 else 0.0
            else:
                x = y = r = 0.0
            
            segments_list.append(win[feat_cols].values.astype(np.float32))
            meta_list.append({
                'timestamp_end': float(end_row['timestamp']),
                'activity': end_row['activity'],
                'phase_pct': float(end_row['phase_pct']),
                'regression_mask': reg_mask,
                'x': float(x), 'y': float(y), 'r': float(r),
                'subject_id': subject_id,
                'speed': speed,
                'cycle_id': int(end_row['cycle_id']),
                'segment_id': i,
                'window_ms': w_ms,
                'trial_id': trial_id,
                'event_code': int(end_row['event_code'])
            })
        
        if not segments_list: continue
        segments = np.stack(segments_list)
        meta_df = pd.DataFrame(meta_list)
        
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('segments', data=segments, compression='gzip')
            for col in meta_df.columns:
                data = meta_df[col].values
                if meta_df[col].dtype == 'object':
                    data = data.astype('S20')
                f.create_dataset(col, data=data)

    print(f"✅ {subject_id} {speed} {trial_id} 完成")

if __name__ == "__main__":
    for f in sorted(RAW_DIR.glob("*.csv")):
        process_one_raw(f)
    print("\n🎉 Walking-Only Phase Preprocessing Stable Version 完成！")