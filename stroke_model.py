import pandas as pd
import numpy as np
import os

# 配置
PREV_WINDOW_NUM = 3
AFTER_WINDOW_NUM = 3

def get_feature_cols(prev_window_num=PREV_WINDOW_NUM, after_window_num=AFTER_WINDOW_NUM):
    # 为了匹配 stroke_predictor.py 的 24 维特征顺序
    # 总特征数 = (prev + after) * 3 * 2 (x, y) = 4 * 3 * 2 = 24
    total_window = prev_window_num + after_window_num
    cols = []
    for axis in ['x', 'y']:
        cols.extend([f'{axis}_custom_diff_{i}' for i in range(total_window)])
        cols.extend([f'{axis}_custom_diff_inv_{i}' for i in range(total_window)])
        cols.extend([f'{axis}_custom_div_{i}' for i in range(total_window)])
    return cols

def to_features(data, prev_window_num=PREV_WINDOW_NUM, after_window_num=AFTER_WINDOW_NUM):
    eps = 1e-15
    data = data.copy()
    
    # 1. 计算基础 lag
    # before
    for i in range(1, prev_window_num + 1):
        data[f'x_lag_{i}'] = data['x'].shift(i)
        data[f'y_lag_{i}'] = data['y'].shift(i)
    # after
    for i in range(1, after_window_num + 1):
        data[f'x_inv_{i}'] = data['x'].shift(-i)
        data[f'y_inv_{i}'] = data['y'].shift(-i)

    # 2. 计算相对于 current 的差值 (diff)
    for axis in ['x', 'y']:
        data[f'{axis}_d_zero'] = 0.0
        # Lag diffs (before)
        for i in range(1, prev_window_num + 1):
            data[f'{axis}_d_b{i}'] = data[f'{axis}_lag_{i}'] - data[axis]
        # Inv diffs (after)
        for i in range(1, after_window_num + 1):
            data[f'{axis}_d_a{i}'] = data[f'{axis}_inv_{i}'] - data[axis]

    # 3. 构造与 stroke_predictor.py 对应的特征列
    # 构造按时间顺序排列的点对应 diff 列名列表: [b2, b1, zero, a1, a2]
    # (注意: b2 是最远过去, b1 是最近过去)
    
    for axis in ['x', 'y']:
        diff_cols_ordered = []
        # Add befores (从大到小: b2, b1)
        for i in range(prev_window_num, 0, -1):
            diff_cols_ordered.append(f'{axis}_d_b{i}')
        
        diff_cols_ordered.append(f'{axis}_d_zero')
        
        # Add afters (从小到大: a1, a2)
        for i in range(1, after_window_num + 1):
            diff_cols_ordered.append(f'{axis}_d_a{i}')
            
        total_points = len(diff_cols_ordered) # 5 for win=2+2
        
        # Diff features mapping
        # Predictor logic: for i in range(1, 5): idx = 5 - 1 - i
        for i in range(1, total_points): 
            target_col = diff_cols_ordered[total_points - 1 - i]
            data[f'{axis}_custom_diff_{i-1}'] = data[target_col]

        # Diff inv features mapping
        # Predictor: lag_pos = positions[i] -> indices 1, 2, 3, 4
        for i in range(1, total_points):
            target_col = diff_cols_ordered[i]
            data[f'{axis}_custom_diff_inv_{i-1}'] = data[target_col]
            
        # Div features
        total_features = total_points - 1
        for i in range(total_features):
            data[f'{axis}_custom_div_{i}'] = data[f'{axis}_custom_diff_{i}'] / (data[f'{axis}_custom_diff_inv_{i}'] + eps)

    # 4. 过滤 NaN
    drop_cols = []
    for i in range(1, prev_window_num + 1):
        drop_cols.append(f'x_lag_{i}')
    for i in range(1, after_window_num + 1):
        drop_cols.append(f'x_inv_{i}')
    
    data = data.dropna(subset=drop_cols + ['x'])
    
    return data

def __add_weight(pd_data, weight_map):   # 为数据添加权重，weight_map是一个字典，key是类别，value是权重
    pd_data["weight"] = pd_data["event_cls"].map(weight_map)
    return pd_data


def load_data(file_path, shuffle=True):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件 {file_path} 不存在。")
    
    # 1. 读取 CSV
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # 2. 过滤掉未核对的数据 (is_checked =！1 的是未标注数据，视为脏数据丢弃)
    if 'is_checked' in df.columns:
        original_len = len(df)
        df = df[df['is_checked'] == 1].copy()
        print(f"Filtered unchecked data: {original_len} -> {len(df)}")
    
    # 3. 解析标签 (event_cls)
    # hit_frames_global 格式可能是 "-1" 或 "49653" 或 "49653,51000"
    def check_is_hit(row):
        hit_str = str(row['hit_frames_global'])
        if hit_str == "-1" or hit_str == "":
            return 0
        hits = hit_str.split(',')
        # 如果当前帧号在击球帧列表中，则为正样本
        return 1 if str(row['frame_index']) in hits else 0
        
    df['event_cls'] = df.apply(check_is_hit, axis=1)
    
    # 4. 特征工程 (必须按 traj_id 分组处理，否则会在不同轨迹交界处产生错误的差分特征)
    # 先按 traj_id 和 frame_index 排序，确保时序正确
    df = df.sort_values(by=['traj_id', 'frame_index'])
    
    resdf = pd.DataFrame()
    
    # 使用 groupby 对每个轨迹单独计算特征
    # 注意：这里会比较耗时，但必须这样做以保证特征准确性
    grouped = df.groupby('traj_id')
    processed_list = []
    
    print("Processing features by trajectory group...")
    for traj_id, group in grouped:
        # 只有当轨迹长度足够计算窗口时才保留
        expected_len = PREV_WINDOW_NUM + AFTER_WINDOW_NUM
        if len(group) > expected_len:
            processed_group = to_features(group)
            processed_list.append(processed_group)
            
    if len(processed_list) > 0:
        resdf = pd.concat(processed_list, ignore_index=True)
    else:
        # 如果没有数据生成特征，可以返回空DataFrame或者抛出异常 warning
        print("Warning: 没有足够的数据生成特征，请检查 traj_id 分组或窗口大小配置。")
        return pd.DataFrame()

    # 6. 添加权重
    resdf = __add_weight(resdf, {1: 40, 0: 1})
    
    if shuffle:
        resdf = resdf.sample(frac=1, random_state=42).reset_index(drop=True)
        
    return resdf
