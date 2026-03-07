# ============================== preprocess_gait.py ==============================
import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, medfilt
from pathlib import Path
import h5py  # 用于高效保存 segments（时序窗口）

# ============================ 配置区（请修改） ============================
RAW_DIR = Path("data/raw")                    # 您的 raw csv 存放目录
OUTPUT_BASE = Path("data")
TARGET_FS = 200                               # 与论文一致（Hz）
WINDOW_MS_OPTIONS = [150, 600]                # 支持两种窗口长度
HEEL_THRESHOLD = 300                          # FSR_heel 阈值（根据您的 FSR 实际波形微调）
MIN_CYCLE_SAMPLES = 300                       # 最小周期长度（防止误检）
SUBJECT_MAP = {                               # 文件名 → subject_id（根据您的实际文件名修改）
    "right_001": "s1", "right_002": "s2",
    "right_003": "s3", "right_004": "s4",
    # 增加更多映射...
}

# ============================ 辅助函数 ============================
def load_raw(file_path: Path) -> pd.DataFrame:
    """Teensy 输出无表头，13列"""
    cols = ['knee_angle', 'knee_ang_vel', 'imu_angle', 'ax', 'ay', 'az',
            'gx', 'gy', 'gz', 'fsr_toe', 'fsr_heel', 'timestamp', 'event_code']
    df = pd.read_csv(file_path, header=None, names=cols)
    df['timestamp'] = df['timestamp'].astype(float)
    return df

def align_and_resample(df: pd.DataFrame) -> pd.DataFrame:
    """1. 时间对齐最先执行（必须在所有 event 检测之前）"""
    # 1. 创建 imu_sample_idx（识别新 IMU 帧）
    imu_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    change_mask = (df[imu_cols].diff().abs().sum(axis=1) > 0.01)  # 浮点容差
    df['imu_sample_idx'] = change_mask.cumsum()
    
    # 2. 去重（保留新 IMU 帧的最新行，其他传感器保持 1kHz 高分辨率）
    df = df.loc[df['imu_sample_idx'].diff() != 0].copy()  # 保留更新时刻
    
    # 3. 重采样到 TARGET_FS（线性插值连续信号，最近邻 IMU）
    df = df.set_index('timestamp')
    new_index = np.arange(df.index[0], df.index[-1], 1.0 / TARGET_FS)
    df_resampled = df.reindex(new_index).interpolate(method='linear')
    df_resampled[imu_cols] = df_resampled[imu_cols].ffill()  # IMU 保持最后值
    df_resampled = df_resampled.reset_index().rename(columns={'index': 'timestamp'})
    return df_resampled

def detect_heel_strikes(df: pd.DataFrame) -> np.ndarray:
    """FSR 去抖 + heel-strike 检测"""
    fsr = medfilt(df['fsr_heel'].values, kernel_size=5)  # 去抖
    # 假设 heel strike 时 FSR_heel 值下降（根据您的接线调整方向）
    peaks, _ = find_peaks(-fsr, distance=MIN_CYCLE_SAMPLES, prominence=100, height=-HEEL_THRESHOLD)
    return peaks

# ============================ 主流程 ============================
def process_one_raw(raw_path: Path, subject_id: str, activity: str, speed: str):
    print(f"处理 {raw_path.name} → {subject_id} | {activity}-{speed}")
    
    df = load_raw(raw_path)
    df = align_and_resample(df)                     # ← 时间对齐最先
    
    # ====================== 1. Cycles ======================
    cycle_dir = OUTPUT_BASE / "cycles" / f"{subject_id}_walking-level-{speed}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    
    hs_idx = detect_heel_strikes(df)
    cycles = []
    for i in range(len(hs_idx) - 1):
        start, end = hs_idx[i], hs_idx[i + 1]
        if end - start < MIN_CYCLE_SAMPLES:
            continue
        cycle_df = df.iloc[start:end].copy().reset_index(drop=True)
        
        # phase_pct 线性生成
        cycle_df['phase_pct'] = np.linspace(0, 100, len(cycle_df))
        
        # 元数据
        cycle_df['subject_id'] = subject_id
        cycle_df['activity'] = activity
        cycle_df['speed'] = speed
        cycle_df['cycle_id'] = i + 1
        cycle_df['start_time'] = cycle_df['timestamp'].iloc[0]
        cycle_df['end_time'] = cycle_df['timestamp'].iloc[-1]
        
        # 保存单个 cycle（用于质量检查）
        cycle_path = cycle_dir / f"{subject_id}_walking-level-{speed}_cycle_{i+1:03d}.csv"
        cycle_df.to_csv(cycle_path, index=False)
        cycles.append(cycle_df)
    
    # ====================== 2. phase_summary ======================
    if cycles:
        summary_df = pd.concat(cycles, ignore_index=True)
        summary_cols = ['subject_id', 'activity', 'speed', 'cycle_id', 'phase_pct',
                        'knee_angle', 'knee_ang_vel', 'ax', 'ay', 'az', 'gx', 'gy', 'gz',
                        'fsr_toe', 'fsr_heel']
        summary_df = summary_df[summary_cols]
        
        summary_path = OUTPUT_BASE / "phase_summary" / f"phase_summary_walking-level-{speed}.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        # 追加模式（多 subject 合并）
        if summary_path.exists():
            summary_df.to_csv(summary_path, mode='a', header=False, index=False)
        else:
            summary_df.to_csv(summary_path, index=False)
    
    # ====================== 3. segments（直接从 raw_aligned 生成） ======================
    for window_ms in WINDOW_MS_OPTIONS:
        seg_dir = OUTPUT_BASE / "segments" / f"window_{window_ms}ms"
        seg_dir.mkdir(parents=True, exist_ok=True)
        
        window_samples = int(window_ms * TARGET_FS / 1000)
        feature_cols = ['knee_angle', 'knee_ang_vel', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
        
        # 使用 h5py 高效保存（每个 subject-speed 一个组）
        h5_path = seg_dir / f"{subject_id}_walking-level-{speed}_segments.h5"
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('segments', shape=(0, window_samples, 8), maxshape=(None, window_samples, 8), dtype='float32')
            f.create_dataset('timestamp_end', shape=(0,), maxshape=(None,), dtype='float32')
            f.create_dataset('activity', shape=(0,), maxshape=(None,), dtype='S20')
            f.create_dataset('phase_pct', shape=(0,), maxshape=(None,), dtype='float32')
            f.create_dataset('subject_id', shape=(0,), maxshape=(None,), dtype='S10')
            f.create_dataset('speed', shape=(0,), maxshape=(None,), dtype='S10')
            
            n = len(df) - window_samples
            for i in range(n):
                window = df.iloc[i:i + window_samples]
                seg = window[feature_cols].values.astype(np.float32)
                
                # 标签取窗口末尾时刻
                end_ts = window['timestamp'].iloc[-1]
                act = window['activity'].iloc[-1] if 'activity' in window else activity
                
                # 追加写入
                f['segments'].resize((len(f['segments']) + 1, window_samples, 8))
                f['segments'][-1] = seg
                
                f['timestamp_end'].resize((len(f['timestamp_end']) + 1,))
                f['timestamp_end'][-1] = end_ts
                
                f['activity'].resize((len(f['activity']) + 1,))
                f['activity'][-1] = act.encode()
                
                f['phase_pct'].resize((len(f['phase_pct']) + 1,))
                f['phase_pct'][-1] = window['phase_pct'].iloc[-1] if 'phase_pct' in window else 0.0
                
                f['subject_id'].resize((len(f['subject_id']) + 1,))
                f['subject_id'][-1] = subject_id.encode()
                
                f['speed'].resize((len(f['speed']) + 1,))
                f['speed'][-1] = speed.encode()

# ============================ 批量执行 ============================
if __name__ == "__main__":
    for raw_file in RAW_DIR.glob("*.csv"):
        filename = raw_file.stem
        # 从文件名提取 subject（根据您的实际命名规则修改）
        subject_id = SUBJECT_MAP.get(filename.split('_')[-1], "s1")  
        
        # 根据协议手动映射 activity / speed（或从 event_code 自动拆分）
        # 示例：您可在这里根据文件名或 event_code=1 的时刻拆分不同速度段
        activity = "walking"
        speed = "1kmh" if "1km" in filename.lower() else "4kmh" if "4km" in filename.lower() else "6kmh"
        
        process_one_raw(raw_file, subject_id, activity, speed)
    
    print("✅ 前处理完成！")
    print("   - cycles/     : 用于质量检查")
    print("   - phase_summary/ : 用于 phase 可视化（每个速度一个文件）")
    print("   - segments/   : CNN 输入（h5 格式，时序窗口）")