import pandas as pd
import numpy as np
import os

# 配置
PREV_WINDOW_NUM = 2
AFTER_WINDOW_NUM = 2

def get_feature_cols(prev_window_num=PREV_WINDOW_NUM, after_window_num=AFTER_WINDOW_NUM):
    colnames_x = ['x_diff_{}'.format(i) for i in range(1, prev_window_num)] + \
                ['x_diff_inv_{}'.format(i) for i in range(1, after_window_num)] + \
                ["x_div_{}".format(i) for i in range(1, after_window_num)] #+ \
                #["x"]
    colnames_y = ['y_diff_{}'.format(i) for i in range(1, prev_window_num)] + \
                    ['y_diff_inv_{}'.format(i) for i in range(1, after_window_num)] + \
                    ["y_div_{}".format(i) for i in range(1, after_window_num)] #+ \
                    # ["y"]
    colnames = colnames_x + colnames_y #+ ["coord"]
    return colnames

def to_features(data, prev_window_num=PREV_WINDOW_NUM, after_window_num=AFTER_WINDOW_NUM):
    eps = 1e-15  # 防止除零错误
    data = data.copy()  # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    for i in range(1, prev_window_num):
        data.loc[:, 'x_lag_{}'.format(i)] = data['x'].shift(i)  # 创建一个新列，存储Y坐标的滞后值（即前几个时间点的Y坐标值）  data.loc[:, column_name]表示选择所有行和指定列,   .shift(i)：将这一列的数据向下移动i行
        data.loc[:, 'y_lag_{}'.format(i)] = data['y'].shift(i)
        data.loc[:, 'x_diff_{}'.format(i)] = data['x_lag_{}'.format(i)] - data['x']   # 计算当前点与滞后点的X坐标差值
        data.loc[:, 'y_diff_{}'.format(i)] = data['y_lag_{}'.format(i)] - data['y']


    for i in range(1, after_window_num):
        data.loc[:, 'x_lag_inv_{}'.format(i)] = data['x'].shift(-i)   # data['x'].shift(-i)：向上移动i行，获取未来的值    x_lag_inv_i, y_lag_inv_i: 存储未来i个时间步长的坐标值
        data.loc[:, 'y_lag_inv_{}'.format(i)] = data['y'].shift(-i) 
        data.loc[:, 'x_diff_inv_{}'.format(i)] = data['x_lag_inv_{}'.format(i)] - data['x']        # x_lag_inv_i, y_lag_inv_i: 存储未来i个时间步长的坐标值，利用未来信息（仅在特征工程中使用，实时预测时不可用）
        data.loc[:, 'y_diff_inv_{}'.format(i)] = data['y_lag_inv_{}'.format(i)] - data['y']


    for i in range(1, after_window_num):
        data.loc[:, 'x_div_{}'.format(i)] = data['x_diff_{}'.format(i)]/(data['x_diff_inv_{}'.format(i)] + eps)    # （过去坐标 - 当前坐标）/ （未来坐标 - 当前坐标）
        data.loc[:, 'y_div_{}'.format(i)] = data['y_diff_{}'.format(i)]/(data['y_diff_inv_{}'.format(i)] + eps)

    for i in range(1, prev_window_num):
        data = data[data['x_lag_{}'.format(i)].notna()]     #  保留x_lag_i列中非空（not null and not NaN）的行，移除由于shift操作产生的空值行（因为shift操作会在开始或结尾产生NaN值）
        
    for i in range(1, after_window_num):
        data = data[data['x_lag_inv_{}'.format(i)].notna()]
    
    data = data[data['x'].notna()] 
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
