import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# ================= 1. 配置区 (Configuration) =================
# 你的 Log 路径 (请确认路径正确)
INPUT_CSV = "output/parsed_a.csv"
# 提取出的数据保存到这里
OUTPUT_CSV = "output/training_segment.csv"

# 截取时长(毫秒) 自定义
CLIP_DURATION_MS = 6000000 

# ================= 2. 核心清洗逻辑 =================
def interpolate_trajectory(group):
    """
    [数据清洗核心]
    轨迹插值补全：修复 Log 中因丢帧导致的不连续。
    如果不补全，打标工具播放时会一卡一卡的，模型训练也会报错。
    """
    if group.empty: 
        return group
    
    # 1. 获取这一段轨迹的起止帧号
    min_f = int(group['frame_index'].min())
    max_f = int(group['frame_index'].max())
    
    # 如果帧数是连续的，直接返回，不用折腾
    if len(group) == (max_f - min_f + 1): 
        return group
    
    # 2. 生成一个从 min 到 max 的完整整数序列作为索引
    full_idx = pd.RangeIndex(start=min_f, stop=max_f+1, step=1, name='frame_index')
    
    # 3. 重新索引 (Reindex)，缺的帧会自动变成 NaN
    group = group.drop_duplicates(subset='frame_index').set_index('frame_index')
    group = group.reindex(full_idx)
    
    # 4. 线性插值填充 NaN
    # method='linear': 画一条直线连起来
    group['x'] = group['x'].interpolate(method='linear')
    group['y'] = group['y'].interpolate(method='linear')
    group['timestamp'] = group['timestamp'].interpolate(method='linear')
    
    # ID 这种东西不能插值，直接用前一个的值填充 (Forward Fill)
    group['traj_id'] = group['traj_id'].fillna(method='ffill').fillna(method='bfill')
    
    return group.reset_index()

# ================= 3. 主程序 =================
def main():
    print(" [Step 1] 正在准备训练数据片段...")
    
    if not os.path.exists(INPUT_CSV):
        print(f" 错误：找不到文件 {INPUT_CSV}")
        return

    # 1. 读取完整 Log
    df = pd.read_csv(INPUT_CSV)
    print(f"📚 原始数据总行数: {len(df)}")
    
    # 2. 计算时间范围
    min_ts = df['timestamp'].min()
    max_ts = df['timestamp'].max()
    total_duration_min = (max_ts - min_ts) / 1000 / 60
    print(f"⏱️  总时长: {total_duration_min:.1f} 分钟")
    
    # 3. 截取最中间的一段 (Middle Segment)
    # 比如总长 240 分钟，我们取第 115-125 分钟
    mid_ts = min_ts + (max_ts - min_ts) / 2
    start_ts = mid_ts - (CLIP_DURATION_MS / 2)
    end_ts = mid_ts + (CLIP_DURATION_MS / 2)
    
    print(f"  截取范围: 中间 {CLIP_DURATION_MS/1000/60:.1f} 分钟")
    
    # 过滤数据
    sub_df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)].copy()
    print(f"   -> 截取后数据量: {len(sub_df)} 行")
    
    if len(sub_df) == 0:
        print(" 截取为空！可能是时间戳有问题。")
        return

    # 4. 按轨迹分组并清洗
    print(" 正在清洗轨迹 (插值补全丢帧)...")
    clean_rows = []
    
    # 按 traj_id 分组
    grouped = sub_df.groupby('traj_id')
    
    for _, group in tqdm(grouped):
        # 过滤掉太短的噪点 (比如只闪现了 1-2 帧的误检)
        if len(group) < 5: continue 
        
        # 插值补全
        clean_group = interpolate_trajectory(group)
        
        # 添加打标用的空列
        clean_group['is_hit'] = 0          # 0:没打, 1:打了
        clean_group['hit_frame_global'] = -1 # 击球发生的绝对帧号
        clean_group['is_checked'] = 0      # 0:未检查, 1:已检查
        
        clean_rows.append(clean_group)
        
    if not clean_rows:
        print("  警告： 这段时间内没有有效长轨迹。")
        return
        
    # 5. 合并并保存
    final_df = pd.concat(clean_rows)
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n 准备完成！")
    print(f" 输出文件: {OUTPUT_CSV}")
    print(f" 包含 {len(clean_rows)} 条独立的轨迹。")

if __name__ == "__main__":
    main()
