import pandas as pd
import numpy as np
from pathlib import Path

summary_dir = Path("data/phase_summary")
binned_dir = Path("data/phase_binned_summary")
binned_dir.mkdir(parents=True, exist_ok=True)

for f in summary_dir.glob("*.csv"):
    df = pd.read_csv(f)
    df = df[df['cycle_id'] > 0].copy()
    if df.empty:
        print(f"⚠️ {f.name} 无 walking 数据，跳过")
        continue
    bins = np.arange(0, 101, 5)
    df['phase_bin'] = pd.cut(df['phase_pct'], bins, labels=bins[:-1], include_lowest=True)
    agg = df.groupby('phase_bin').agg({
        'knee_angle': ['mean','std','count'],
        'knee_ang_vel': ['mean','std','count'],
        'ax': ['mean','std','count'], 'ay': ['mean','std','count'], 'az': ['mean','std','count'],
        'gx': ['mean','std','count'], 'gy': ['mean','std','count'], 'gz': ['mean','std','count'],
        'fsr_heel': ['mean','std','count'], 'fsr_toe': ['mean','std','count']
    })
    agg = agg.fillna(0)
    agg.to_csv(binned_dir / f"binned_{f.name}")
    print(f"✅ 已生成 phase-binned: {f.name}")

# TODO: 跨文件总聚合（未来论文图用）
# all_df = pd.concat([pd.read_csv(f) for f in summary_dir.glob("*.csv") if "walking" in f.name])
# ... 按 speed / subject 再聚合