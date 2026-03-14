import pandas as pd
import numpy as np
from pathlib import Path

summary_dir = Path("data/phase_summary")
binned_dir = Path("data/phase_binned_summary")
binned_dir.mkdir(parents=True, exist_ok=True)

for f in summary_dir.glob("phase_summary_walking_*.csv"):
    df = pd.read_csv(f)

    if 'phase_valid_mask' in df.columns:
        df = df[df['phase_valid_mask'] == 1].copy()
    else:
        df = df.dropna(subset=['phase_pct']).copy()

    if df.empty:
        print(f"⚠️ {f.name} 无有效 walking phase 数据，跳过")
        continue

    bins = np.arange(0.0, 1.0001, 0.05)
    labels = bins[:-1]

    df['phase_bin'] = pd.cut(
        df['phase_pct'],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False
    )

    agg = df.groupby('phase_bin', observed=False).agg({
        'knee_angle': ['mean', 'std', 'count'],
        'knee_ang_vel': ['mean', 'std', 'count'],
        'ax': ['mean', 'std', 'count'],
        'ay': ['mean', 'std', 'count'],
        'az': ['mean', 'std', 'count'],
        'gx': ['mean', 'std', 'count'],
        'gy': ['mean', 'std', 'count'],
        'gz': ['mean', 'std', 'count'],
        'heel_fsr': ['mean', 'std', 'count'],
        'toe_fsr': ['mean', 'std', 'count'],
        'cycle_duration_s': ['mean', 'std']
    }).fillna(0)

    agg.to_csv(binned_dir / f"binned_{f.name}")
    print(f"✅ 已生成 phase-binned: {f.name}")