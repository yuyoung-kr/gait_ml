# ============================== preprocess_gait_v9_final.py ==============================
import pandas as pd
import numpy as np
from scipy.signal import medfilt
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import warnings

# ============================ 配置 ============================
RAW_DIR = Path("data/raw")
OUTPUT_BASE = Path("data")
METADATA_FILE = Path("metadata.csv")

TARGET_FS = 200.0
WINDOW_MS_LIST = [150, 600]
STRIDE_SAMPLES = 10

MIN_CYCLE_DURATION_S = 0.4

# FSR filtering / hysteresis / debounce
FSR_MEDIAN_KERNEL = 7
FSR_MA_WIN = 5

HEEL_ON_THRESHOLD = 350
HEEL_OFF_THRESHOLD = 250
TOE_ON_THRESHOLD = 350
TOE_OFF_THRESHOLD = 250

MIN_CONTACT_SAMPLES = 4
MIN_SWING_SAMPLES = 4

# region definition
STEADY_BUFFER_S = 2.0
TAIL_TRIM_S = 3.0

def load_metadata():
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"请创建 {METADATA_FILE}")

    meta = pd.read_csv(METADATA_FILE)
    meta['filename'] = (
        meta['filename']
        .astype(str)
        .str.lower()
        .str.replace('.csv', '', regex=False)
    )

    required_cols = [
        'subject_id',
        'split',
        'protocol',
        'activity_nominal',
        'condition_value',
        'condition_unit',
        'trial_id',
        'filename'
    ]
    missing = [c for c in required_cols if c not in meta.columns]
    if missing:
        raise ValueError(f"metadata.csv 缺少列: {missing}")

    if meta['filename'].duplicated().any():
        raise ValueError("metadata.csv 中 filename 重复！")

    if meta[required_cols].isnull().any().any():
        raise ValueError("metadata.csv 中存在空字段！")

    valid_split = {'train', 'eval'}
    valid_protocol = {'training', 'steady_speed', 'activity_changing', 'speed_varying'}
    valid_activity = {'standing', 'walking', 'stair-ascent', 'stair-descent'}
    valid_unit = {'kmh', 'spm', 'none'}

    if not meta['split'].isin(valid_split).all():
        raise ValueError(f"split 只能取 {valid_split}")
    if not meta['protocol'].isin(valid_protocol).all():
        raise ValueError(f"protocol 只能取 {valid_protocol}")
    if not meta['activity_nominal'].isin(valid_activity).all():
        raise ValueError(f"activity_nominal 只能取 {valid_activity}")
    if not meta['condition_unit'].isin(valid_unit).all():
        raise ValueError(f"condition_unit 只能取 {valid_unit}")

    return meta.set_index('filename')

METADATA = load_metadata()

def get_metadata(filename: str):
    stem = filename.lower().replace(".csv", "")
    if stem not in METADATA.index:
        raise ValueError(f"❌ 文件名未在 metadata.csv 中: {filename}")
    row = METADATA.loc[stem]
    return row.to_dict()

# ============================ 异常检查 + 修复 + 复检 ============================
def check_and_fix_anomalies(df, subject_id, activity_nominal, trial_id):
    # timestamp 先处理
    if df['timestamp'].isna().any():
        warnings.warn(f"[{subject_id} {activity_nominal} {trial_id}] NaN 时间戳 → 已填充")
        df['timestamp'] = df['timestamp'].ffill().bfill()

    # 强制排序
    if (df['timestamp'].diff() < 0).any():
        warnings.warn(f"[{subject_id} {activity_nominal} {trial_id}] 时间倒退 → 已排序")
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 强制去重
    if df['timestamp'].duplicated().any():
        warnings.warn(f"[{subject_id} {activity_nominal} {trial_id}] 重复时间戳 → 已去重")
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

    # sensor NaN 修复
    sensor_cols = [
        'knee_angle', 'knee_ang_vel', 'imu_angle',
        'ax', 'ay', 'az', 'gx', 'gy', 'gz',
        'fsr_toe', 'fsr_heel'
    ]
    for col in sensor_cols:
        if df[col].isna().any():
            df[col] = df[col].interpolate().bfill().ffill()

    # imu_sample_idx 异常提示
    if 'imu_sample_idx' in df.columns and not df['imu_sample_idx'].isna().all():
        if (df['imu_sample_idx'].diff().fillna(0) < 0).any():
            warnings.warn(f"[{subject_id} {activity_nominal} {trial_id}] imu_sample_idx 存在回退")

    # 复检
    for col in sensor_cols + ['timestamp']:
        if df[col].isna().any():
            raise ValueError(f"❌ {col} 修复后仍有 NaN，跳过文件 {subject_id}_{activity_nominal}")

    print(f"✅ 异常检查+修复+复检完成: {subject_id} {activity_nominal} {trial_id}")
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

def extract_valid_imu_frames(df):
    df = df.copy()
    imu_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

    if df['imu_sample_idx'].isna().all():
        changed = df[imu_cols].diff().abs().max(axis=1).fillna(1.0) > 0.001
        valid_mask = changed.copy()
        valid_mask.iloc[0] = True
    else:
        idx_change = df['imu_sample_idx'].ne(df['imu_sample_idx'].shift(1))
        valid_mask = idx_change.copy()
        valid_mask.iloc[0] = True

    df_imu = df.loc[valid_mask, ['timestamp'] + imu_cols].copy()
    df_imu = df_imu.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    return df_imu

def extract_event_markers(df):
    event_series = df['event_code'].fillna(0).astype(float)

    # 每次按 MARK 的“起点” = 从 0 跳到非 0 的上升沿
    press_mask = (event_series > 0) & (event_series.shift(1, fill_value=0) <= 0)
    press_times = df.loc[press_mask, 'timestamp'].astype(float).to_list()

    # 两两配对: (event1, event2), (event1, event2), ...
    pairs = []
    n_pairs = len(press_times) // 2
    for k in range(n_pairs):
        t1 = press_times[2 * k]
        t2 = press_times[2 * k + 1]
        if t2 > t1:
            pairs.append({
                'pair_id': k + 1,
                'event1_time': float(t1),
                'event2_time': float(t2),
            })

    # 如果是奇数个 press，最后一个没有配对，先记 warning 信息
    unmatched_event1_time = np.nan
    if len(press_times) % 2 == 1:
        unmatched_event1_time = float(press_times[-1])

    markers = {
        'press_times': press_times,     # 所有按键时间
        'pairs': pairs,                 # 多组 (event1, event2)
        'n_pairs': len(pairs),
        'first_event1_time': pairs[0]['event1_time'] if len(pairs) > 0 else np.nan,
        'first_event2_time': pairs[0]['event2_time'] if len(pairs) > 0 else np.nan,
        'unmatched_event1_time': unmatched_event1_time,
    }
    return markers

def align_and_resample(df, markers):
    df = df.copy()

    t_start, t_end = df['timestamp'].min(), df['timestamp'].max()
    new_index = np.arange(t_start, t_end + 1e-9, 1.0 / TARGET_FS)

    df_res = pd.DataFrame({'timestamp': new_index})

    # 1) IMU 只对有效帧插值
    df_imu = extract_valid_imu_frames(df)
    imu_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    for col in imu_cols:
        df_res[col] = np.interp(new_index, df_imu['timestamp'], df_imu[col])

    # 2) 膝角 / 膝角速度 / FSR 用全时基插值
    other_cols = ['knee_angle', 'knee_ang_vel', 'imu_angle', 'fsr_toe', 'fsr_heel']
    for col in other_cols:
        df_res[col] = np.interp(new_index, df['timestamp'], df[col])

    # 3) event marker 映射到统一时基：支持多组 pair
    df_res['event_pulse'] = 0
    df_res['event1_flag'] = 0
    df_res['event2_flag'] = 0

    # 每个采样点所属哪一组 pair；0 表示不属于任何 pair
    df_res['event_pair_id'] = 0

    # 记录“当前样本最近/当前所属”的事件对时间
    df_res['current_event1_time'] = np.nan
    df_res['current_event2_time'] = np.nan

    # 先把所有 press 映射成 event_pulse
    for t_evt in markers['press_times']:
        idx = int(np.argmin(np.abs(df_res['timestamp'].values - t_evt)))
        df_res.loc[idx, 'event_pulse'] = 1

    # 再把每一组 pair 的 event1 / event2 映射到统一时基
    for pair in markers['pairs']:
        pair_id = pair['pair_id']
        t1 = pair['event1_time']
        t2 = pair['event2_time']

        idx1 = int(np.argmin(np.abs(df_res['timestamp'].values - t1)))
        idx2 = int(np.argmin(np.abs(df_res['timestamp'].values - t2)))

        df_res.loc[idx1, 'event1_flag'] = 1
        df_res.loc[idx2, 'event2_flag'] = 1

        # 在 event1 ~ event2 之间标记当前 pair_id
        pair_mask = (df_res['timestamp'] >= t1) & (df_res['timestamp'] < t2)
        df_res.loc[pair_mask, 'event_pair_id'] = pair_id

        # steady 段也继续保留当前 pair 的时间，直到下一个 pair 覆盖
        after_t1_mask = df_res['timestamp'] >= t1
        df_res.loc[after_t1_mask, 'current_event1_time'] = t1
        df_res.loc[after_t1_mask, 'current_event2_time'] = t2

    # 兼容旧字段：仅保留第一组，防止下游代码直接报错
    df_res['event1_time'] = markers['first_event1_time']
    df_res['event2_time'] = markers['first_event2_time']

    return df_res

def moving_average(x, win):
    if win <= 1:
        return x.copy()
    return pd.Series(x).rolling(win, center=True, min_periods=1).mean().values

def hysteresis_binarize(x, on_th, off_th):
    state = np.zeros(len(x), dtype=np.int8)
    cur = 0
    for i, v in enumerate(x):
        if cur == 0 and v >= on_th:
            cur = 1
        elif cur == 1 and v <= off_th:
            cur = 0
        state[i] = cur
    return state

def debounce_binary_state(x, min_on=4, min_off=4):
    x = np.asarray(x, dtype=np.int8).copy()
    n = len(x)
    if n == 0:
        return x

    start = 0
    while start < n:
        val = x[start]
        end = start + 1
        while end < n and x[end] == val:
            end += 1
        seg_len = end - start

        if val == 1 and seg_len < min_on:
            x[start:end] = 0
        elif val == 0 and seg_len < min_off:
            x[start:end] = 1

        start = end
    return x

def detect_heel_strikes(df):
    heel_raw = df['fsr_heel'].values.astype(float)
    toe_raw = df['fsr_toe'].values.astype(float)

    heel_filt = moving_average(medfilt(heel_raw, FSR_MEDIAN_KERNEL), FSR_MA_WIN)
    toe_filt = moving_average(medfilt(toe_raw, FSR_MEDIAN_KERNEL), FSR_MA_WIN)

    heel_contact = hysteresis_binarize(heel_filt, HEEL_ON_THRESHOLD, HEEL_OFF_THRESHOLD)
    toe_contact = hysteresis_binarize(toe_filt, TOE_ON_THRESHOLD, TOE_OFF_THRESHOLD)

    heel_contact = debounce_binary_state(
        heel_contact,
        min_on=MIN_CONTACT_SAMPLES,
        min_off=MIN_SWING_SAMPLES
    )
    toe_contact = debounce_binary_state(
        toe_contact,
        min_on=MIN_CONTACT_SAMPLES,
        min_off=MIN_SWING_SAMPLES
    )

    hs_candidates = np.where((heel_contact[1:] == 1) & (heel_contact[:-1] == 0))[0] + 1

    # refractory: 两次 HS 间隔必须 > MIN_CYCLE_DURATION_S
    hs_idx = []
    min_gap = int(MIN_CYCLE_DURATION_S * TARGET_FS)
    for idx in hs_candidates:
        if not hs_idx or (idx - hs_idx[-1]) >= min_gap:
            hs_idx.append(idx)

    aux = {
        'heel_filt': heel_filt,
        'toe_filt': toe_filt,
        'heel_contact': heel_contact,
        'toe_contact': toe_contact,
    }
    return np.asarray(hs_idx, dtype=int), aux

def save_heelstrike_qa(df, hs_idx, aux, subject_id, condition_tag, trial_id):
    plt.figure(figsize=(12, 5))
    plt.plot(df['timestamp'], df['fsr_heel'], label='FSR Heel raw', alpha=0.5)
    plt.plot(df['timestamp'], aux['heel_filt'], label='FSR Heel filt', linewidth=1.5)
    plt.plot(df['timestamp'], aux['heel_contact'] * np.nanmax(aux['heel_filt']), label='Heel contact')
    if len(hs_idx) > 0:
        plt.scatter(df.iloc[hs_idx]['timestamp'], aux['heel_filt'][hs_idx],
                    color='red', s=25, label='Heel-strike')
    plt.title(f'Heel-Strike QA: {subject_id} {condition_tag} {trial_id} | {len(hs_idx)} strikes')
    plt.legend()
    (OUTPUT_BASE / "qa_heelstrike").mkdir(parents=True, exist_ok=True)
    plt.savefig(
        OUTPUT_BASE / "qa_heelstrike" / f"{subject_id}_{condition_tag}_{trial_id}_qa.png",
        dpi=150, bbox_inches='tight'
    )
    plt.close()

def assign_activity_regions(df_aligned, meta, markers):
    df = df_aligned.copy()

    # 默认区间标签
    df['region_label'] = 'pre_start'
    df['is_transition'] = 0
    df['is_steady'] = 0
    df['is_eval_region'] = 1 if meta['split'] == 'eval' else 0
    df['region_pair_id'] = 0

    t = df['timestamp'].values
    t_end = t[-1]

    protocol = meta['protocol']
    nominal_activity = meta['activity_nominal']

    # ---------- activity 默认值 ----------
    # activity_changing: 默认全段 standing
    # 其他 locomotion 文件: 默认全段 nominal activity
    if nominal_activity == 'standing':
        df['activity'] = 'standing'
    elif protocol == 'activity_changing':
        df['activity'] = 'standing'
    else:
        df['activity'] = nominal_activity

    # standing 文件：全段 steady
    if nominal_activity == 'standing':
        df['region_label'] = 'steady'
        df['is_steady'] = 1
        return df

    # 没有 pair
    if len(markers['pairs']) == 0:
        tail_start = t_end - TAIL_TRIM_S
        if tail_start < t_end:
            tail_mask = t >= tail_start
            df.loc[tail_mask, 'region_label'] = 'tail'
        return df

    # ---------- 逐组 pair 写 region ----------
    for i, pair in enumerate(markers['pairs']):
        pair_id = pair['pair_id']
        t1 = pair['event1_time']
        t2 = pair['event2_time']
        next_t1 = markers['pairs'][i + 1]['event1_time'] if i + 1 < len(markers['pairs']) else np.inf

        # 本组 pair 对应的“目标活动”
        # activity_changing: 奇数 pair 进入 nominal，偶数 pair 回 standing
        # 其他协议: 一律 nominal
        if protocol == 'activity_changing':
            target_activity = nominal_activity if (pair_id % 2 == 1) else 'standing'
        else:
            target_activity = nominal_activity

        # 1) transition: [event1, event2)
        transition_mask = (t >= t1) & (t < t2)
        df.loc[transition_mask, 'region_label'] = 'transition'
        df.loc[transition_mask, 'is_transition'] = 1
        df.loc[transition_mask, 'region_pair_id'] = pair_id
        df.loc[transition_mask, 'activity'] = target_activity

        # 2) steady: [event2 + buffer, min(next_event1, end-tail))
        steady_start = t2 + STEADY_BUFFER_S
        steady_end = min(next_t1, t_end - TAIL_TRIM_S)

        if steady_start < steady_end:
            steady_mask = (t >= steady_start) & (t < steady_end)
            df.loc[steady_mask, 'region_label'] = 'steady'
            df.loc[steady_mask, 'is_steady'] = 1
            df.loc[steady_mask, 'region_pair_id'] = pair_id
            df.loc[steady_mask, 'activity'] = target_activity

    # ---------- tail ----------
    tail_start = t_end - TAIL_TRIM_S
    if tail_start < t_end:
        tail_mask = t >= tail_start
        df.loc[tail_mask, 'region_label'] = 'tail'

        if protocol == 'activity_changing':
            # 最后一段通常应回 standing；若 pair 数为奇数，则说明停在 nominal activity
            final_activity = 'standing' if (len(markers['pairs']) % 2 == 0) else nominal_activity
            df.loc[tail_mask, 'activity'] = final_activity

    return df

def process_one_raw(raw_path: Path):
    meta = get_metadata(raw_path.name)
    subject_id = meta['subject_id']
    trial_id = meta['trial_id']
    activity_nominal = meta['activity_nominal']
    condition_value = meta['condition_value']
    condition_unit = meta['condition_unit']

    if condition_unit == 'none':
        condition_tag = f"{activity_nominal}"
    else:
        condition_tag = f"{activity_nominal}-{condition_value}{condition_unit}"

    print(f"处理 {raw_path.name} → {subject_id} | {condition_tag} | {trial_id}")

    df = load_raw(raw_path)
    df = check_and_fix_anomalies(df, subject_id, activity_nominal, trial_id)

    markers = extract_event_markers(df)

    if not np.isnan(markers['unmatched_event1_time']):
        warnings.warn(
            f"[{subject_id} {activity_nominal} {trial_id}] "
            f"检测到未配对的最后一个 event1，已忽略该不完整 pair"
    )
    df_aligned = align_and_resample(df, markers)
    df_aligned = assign_activity_regions(df_aligned, meta, markers)

    df_aligned['subject_id'] = subject_id
    df_aligned['split'] = meta['split']
    df_aligned['protocol'] = meta['protocol']
    df_aligned['activity_nominal'] = activity_nominal
    df_aligned['condition_value'] = condition_value
    df_aligned['condition_unit'] = condition_unit
    df_aligned['trial_id'] = trial_id

    # raw_aligned
    (OUTPUT_BASE / "raw_aligned").mkdir(parents=True, exist_ok=True)
    aligned_name = f"{subject_id}_{condition_tag}_{trial_id}_aligned.csv"
    df_aligned.to_csv(OUTPUT_BASE / "raw_aligned" / aligned_name, index=False)

    # 初始化 phase / cycle
    df_aligned['phase_frac'] = np.nan
    df_aligned['phase_pct'] = np.nan
    df_aligned['cycle_id'] = 0
    df_aligned['cycle_duration'] = np.nan

    # standing 不做 HS / cycle
    if activity_nominal != 'standing':
        hs_idx, aux = detect_heel_strikes(df_aligned)
        save_heelstrike_qa(df_aligned, hs_idx, aux, subject_id, condition_tag, trial_id)

        cycle_dir = OUTPUT_BASE / "cycles" / condition_tag
        cycle_dir.mkdir(parents=True, exist_ok=True)

        cycle_counter = 0

        for i in range(len(hs_idx) - 1):
            start = hs_idx[i]
            end = hs_idx[i + 1]   # exclusive

            # 当前候选 cycle 的整段数据
            cycle_slice = df_aligned.iloc[start:end].copy()
            if len(cycle_slice) == 0:
                continue

            duration = df_aligned.iloc[end]['timestamp'] - df_aligned.iloc[start]['timestamp']
            if duration < MIN_CYCLE_DURATION_S:
                continue

            n = end - start
            if n <= 0:
                continue

            # ---------- 关键过滤：只保留 locomotion steady bout 内部的 cycle ----------
            same_pair = cycle_slice['region_pair_id'].nunique() == 1
            pair_id_ok = int(cycle_slice['region_pair_id'].iloc[0]) > 0
            all_steady = (cycle_slice['region_label'] == 'steady').all()
            all_nominal = (cycle_slice['activity'] == activity_nominal).all()

            if not (same_pair and pair_id_ok and all_steady and all_nominal):
                continue

            cycle_counter += 1

            phase_frac = np.arange(n, dtype=float) / float(n)
            phase_pct = phase_frac * 100.0

            # 左闭右开 [start, end)
            df_aligned.loc[start:end-1, 'phase_frac'] = phase_frac
            df_aligned.loc[start:end-1, 'phase_pct'] = phase_pct
            df_aligned.loc[start:end-1, 'cycle_id'] = cycle_counter
            df_aligned.loc[start:end-1, 'cycle_duration'] = duration

            cycle_df = cycle_slice.copy()
            cycle_df['start_time'] = df_aligned.iloc[start]['timestamp']
            cycle_df['end_time'] = df_aligned.iloc[end]['timestamp']

            cycle_path = cycle_dir / f"{subject_id}_{trial_id}_cycle_{cycle_counter:03d}.csv"
            cycle_df.to_csv(cycle_path, index=False)

    # phase_summary per-file 中间件
    phase_per_file_dir = OUTPUT_BASE / "phase_summary" / "per_file"
    phase_per_file_dir.mkdir(parents=True, exist_ok=True)

    summary_cols = [
        'subject_id', 'split', 'protocol',
        'activity', 'activity_nominal',
        'condition_value', 'condition_unit',
        'trial_id', 'cycle_id', 'phase_frac', 'phase_pct',
        'knee_angle', 'knee_ang_vel',
        'ax', 'ay', 'az', 'gx', 'gy', 'gz',
        'fsr_toe', 'fsr_heel',
        'timestamp'
    ]
    summary_df = df_aligned[df_aligned['cycle_id'] > 0][summary_cols].copy()
    summary_df.to_csv(
        phase_per_file_dir / f"{subject_id}_{condition_tag}_{trial_id}_phase_rows.csv",
        index=False
    )

    # segments
    feat_cols = ['knee_angle', 'knee_ang_vel', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']

    for w_ms in WINDOW_MS_LIST:
        seg_dir = OUTPUT_BASE / "segments" / f"window_{w_ms}ms"
        seg_dir.mkdir(parents=True, exist_ok=True)

        h5_path = seg_dir / f"{subject_id}_{condition_tag}_{trial_id}_segments.h5"

        window_samples = int(w_ms * TARGET_FS / 1000)
        segments_list, meta_list = [], []

        for i in range(0, len(df_aligned) - window_samples + 1, STRIDE_SAMPLES):
            win = df_aligned.iloc[i:i + window_samples]
            end_row = win.iloc[-1]

            label_time = float(end_row['timestamp'])
            phase_frac = float(end_row['phase_frac']) if pd.notna(end_row['phase_frac']) else 0.0
            cycle_dur = float(end_row['cycle_duration']) if pd.notna(end_row['cycle_duration']) else 0.0

            classifier_mask = int(end_row['region_label'] == 'steady')
            regression_mask = int(
                (activity_nominal != 'standing') and
                (end_row['region_label'] == 'steady') and
                (int(end_row['cycle_id']) > 0)
            )

            if regression_mask:
                x = np.cos(2 * np.pi * phase_frac)
                y = np.sin(2 * np.pi * phase_frac)
                r = 1.0 / cycle_dur if cycle_dur > 0 else 0.0
            else:
                x = y = r = 0.0

            segments_list.append(win[feat_cols].values.astype(np.float32))
            meta_list.append({
                'segment_id': i,
                'window_ms': w_ms,
                'label_time': label_time,
                'activity': str(end_row['activity']),
                'activity_nominal': activity_nominal,
                'phase_frac': phase_frac,
                'phase_pct': float(end_row['phase_pct']) if pd.notna(end_row['phase_pct']) else 0.0,
                'cycle_id': int(end_row['cycle_id']),
                'cycle_duration': cycle_dur,
                'x': float(x),
                'y': float(y),
                'r': float(r),
                'classifier_mask': classifier_mask,
                'regression_mask': regression_mask,
                'region_label': str(end_row['region_label']),
                'is_transition': int(end_row['is_transition']),
                'is_steady': int(end_row['is_steady']),
                'is_eval_region': int(end_row['is_eval_region']),
                'event_pair_id': int(end_row['region_pair_id']) if pd.notna(end_row['region_pair_id']) else 0,
                'event1_time': float(end_row['current_event1_time']) if pd.notna(end_row['current_event1_time']) else np.nan,
                'event2_time': float(end_row['current_event2_time']) if pd.notna(end_row['current_event2_time']) else np.nan,
                'subject_id': subject_id,
                'split': meta['split'],
                'protocol': meta['protocol'],
                'condition_value': condition_value,
                'condition_unit': condition_unit,
                'trial_id': trial_id,
            })

        if not segments_list:
            continue

        segments = np.stack(segments_list)
        meta_df = pd.DataFrame(meta_list)

        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('segments', data=segments, compression='gzip')
            for col in meta_df.columns:
                data = meta_df[col].values
                if meta_df[col].dtype == 'object':
                    max_len = max(len(str(v)) for v in data) if len(data) > 0 else 1
                    data = data.astype(f'S{max_len}')
                f.create_dataset(col, data=data)

    print(f"✅ {subject_id} {condition_tag} {trial_id} 完成")

def aggregate_phase_summary():
    per_file_dir = OUTPUT_BASE / "phase_summary" / "per_file"
    out_dir = OUTPUT_BASE / "phase_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(per_file_dir.glob("*_phase_rows.csv"))
    if not files:
        print("⚠️ 没有 per-file phase rows 可聚合")
        return

    all_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if all_df.empty:
        print("⚠️ phase rows 为空")
        return

    grouped = all_df.groupby(['activity_nominal', 'condition_value', 'condition_unit'], dropna=False)

    for (activity_nominal, condition_value, condition_unit), g in grouped:
        if pd.isna(condition_unit) or condition_unit == 'none':
            out_name = f"phase_summary_{activity_nominal}.csv"
        else:
            out_name = f"phase_summary_{activity_nominal}-{condition_value}{condition_unit}.csv"
        g.to_csv(out_dir / out_name, index=False)

    print("✅ phase_summary 聚合完成")   

if __name__ == "__main__":
    for f in sorted(RAW_DIR.glob("*.csv")):
        process_one_raw(f)

    aggregate_phase_summary()
    print("\n🎉 Protocol-driven preprocessing 完成！")