import pandas as pd
import json
import numpy as np
import os
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score, confusion_matrix

TRAIN_DIR = "data/train"
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"训练数据目录 {TRAIN_DIR} 不存在，请检查路径。")

TEST_DIR = "data/test"
if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"测试数据目录 {TEST_DIR} 不存在，请检查路径。")

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


def __convert_to_dataframe(data, labels_data=[]):
    pd_data = []
    for index, item in enumerate(data):
        item_timestamp = item["timestamp"]
        if item_timestamp in labels_data:
            label = 1
        else:
            label = 0
        label = max(item.get("event_cls", 0), item.get("label_cls", 0), label)
        if item.get("pos", None) is None:
            next_index = -1    # 设置next_index为-1，作为标志位，如果最终next_index仍然是-1，说明没有找到后续的有效位置数据
            for i in range(index + 1, index + 5):     # 从当前索引的下一个开始，往后最多搜索4个位置
                if i >= len(data):     # 如果已经遍历到数据末尾，则跳出循环
                    break
                if data[i].get("pos", None) is not None: 
                    next_index = i
                    break
            if next_index == -1:   # 如果循环结束后next_index仍是-1，说明在接下来的5个数据中都没找到有效位置，则跳过这个点
                continue
            last_data = pd_data[-1] 
            next_item = data[next_index]

            x = (last_data["x"] + next_item["pos"]["x"]) / (next_index - index + 1)   # 计算插值的X坐标，取前后已知点的平均值
            y = (last_data["y"] + next_item["pos"]["y"]) / (next_index - index + 1)   # 计算插值的Y坐标，取前后已知点的平均值
            # if y < 200:  # 只考虑近处的摄像头
            #     label = 0
            pd_data.append({
                "timestamp": item["timestamp"],
                "x": x,
                "y": y,
                "event_cls": label,
                # "coord": 0,  # inserted  "coord": 0 表示这是插入/插值的数据
                "video_file": item.get("video_file", "")
            })
        else:
            y = item["pos"]["y"]
            # if y < 200:  # 只考虑近处的摄像头
            #     label = 0
            pd_data.append({
                "timestamp": item["timestamp"],
                "x": item["pos"]["x"],
                "y": item["pos"]["y"],
                "event_cls": label,
                # "coord": 1,  # real  "coord": 1 表示这是实际观测到的数据点
                "video_file": item.get("video_file", "")
            })
    if len(pd_data) > 0:
        pd_data = pd.DataFrame.from_dict(pd_data)
        pd_data = __add_weight(pd_data, {1: 400, 0: 1})
    return pd_data

def load_data(directories, tag="left", single_view=False, shuffle=True):   # 是否是单视角，如果以后有多视角，设single_view=False。
    for directory in directories:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"目录 {directory} 不存在。")
    resdf = pd.DataFrame()
    for directory in directories:
        if single_view:
            # 单视角：加载 bounce_train.json
            file_path = os.path.join(directory, "bounce_train.json")
        else:
            # 多视角：加载 {tag}_bounce_train.json
            file_path = os.path.join(directory, f"{tag}_bounce_train.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在。")
        
        with open(file_path, "r") as f:
            datalist = [json.loads(line.strip()) for line in f.readlines()]
        
        tracks_data = {}   # 创建字典tracks_data，键为轨迹ID，值为该轨迹的所有数据点 
        for item in datalist:
            track_id = item["track_id"]    # 获取每个的轨迹ID
            if track_id not in tracks_data:   
                tracks_data[track_id] = []    # 如果字典中没有该轨迹ID，则创建一个空列表，用于存储该轨迹的所有数据点
            tracks_data[track_id].append(item)   # 将当前数据点添加到对应轨迹ID的列表中，便于按轨迹处理数据
            
        for track_id, track_data in tracks_data.items():
            track_data = sorted(track_data, key=lambda x: x["timestamp"])
            tmp = __convert_to_dataframe(track_data)
            if len(tmp) > 0:
                video_file = tmp["video_file"].iloc[0] if len(tmp) > 0 and "video_file" in tmp.columns else ""
                tmp["source_video"] = os.path.join(directory, "video", video_file).replace("\\", "/")  # 统一路径分隔符
                resdf = pd.concat([resdf, to_features(tmp)], ignore_index=True)
    if shuffle:
        resdf = resdf.sample(frac=1, random_state=42).reset_index(drop=True)
    return resdf

def train(train_data, test_data):
    if train_data["event_cls"].nunique() < 2:  # 统计event_cls列中唯一值的数量，如果小于2，则说明只有单一类别，即没有正样本
        raise ValueError("训练集中只有单一类别（event_cls 全为同一值）。请检查 bounce_train.json 是否包含正样本，或重新生成标注数据。")
    
    catboost_regressor = CatBoostRegressor(iterations=3000, depth=3, learning_rate=0.1, loss_function='RMSE')
    catboost_regressor.fit(
        train_data[get_feature_cols(PREV_WINDOW_NUM, AFTER_WINDOW_NUM)],  # 训练特征
        train_data['event_cls'],                                         # 训练标签
        eval_set=(test_data[get_feature_cols(PREV_WINDOW_NUM, AFTER_WINDOW_NUM)], test_data['event_cls']),  # 验证集
        use_best_model=True,                                             # 使用最佳模型
        sample_weight=train_data['weight'],                              # 样本权重
        early_stopping_rounds=100,                                    # 早停轮数（注释掉了）
    )
    return catboost_regressor


def evaluate(train_data, test_data, catboost_regressor):
    test_data["pred"] = catboost_regressor.predict(test_data[get_feature_cols(PREV_WINDOW_NUM, AFTER_WINDOW_NUM)])
    
    # 存储每个阈值的指标
    thresholds = []
    accuracies = []
    recalls = []
    precisions = []
    f1_scores = []
    
    for threshold in np.arange(0.1, 1, 0.1):
        print(f'===> threshold: {threshold}')

        # 使用 sklearn 计算混淆矩阵
        pred_labels = (test_data["pred"] > threshold).astype(int)
        cm = confusion_matrix(test_data['event_cls'], pred_labels)
        tn, fp, fn, tp = cm.ravel()  # [[tn, fp], [fn, tp]]
        
        print(f'tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}, total: {tn + tp + fn + fp}')

        acc = (tn + tp) / (tn + tp + fn + fp) if (tn + tp + fn + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f'accuracy: {acc}, recall: {recall}, precision: {precision}, f1: {f1}')
        
        thresholds.append(threshold)
        accuracies.append(acc)
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)

    # 选择最佳阈值（最大化F1-score）
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print(f'Best threshold: {best_threshold} with F1: {f1_scores[best_idx]}')

    print("roc", roc_auc_score(test_data['event_cls'], test_data['pred']))
    
    return best_threshold


def main():
    # 获取训练和测试目录
    train_dirs = [os.path.join(TRAIN_DIR, d) for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d)) and d.startswith("match")]
    test_dirs = [os.path.join(TEST_DIR, d) for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d)) and d.startswith("match")]
    
    # 加载训练和测试数据
    train_data = load_data(train_dirs, single_view=True, shuffle=True)  # 训练集shuffle
    test_data = load_data(test_dirs, single_view=True, shuffle=False)   # 测试集不shuffle，保持时序
    
    print(f"Train data shape: {train_data.shape}, positive samples: {len(train_data[train_data['event_cls'] == 1])}")
    print(f"Test data shape: {test_data.shape}, positive samples: {len(test_data[test_data['event_cls'] == 1])}")

    catboost_regressor = train(train_data, test_data)
    catboost_regressor.save_model("stroke_model.cbm")
    best_threshold = evaluate(train_data, test_data, catboost_regressor)
    
    return best_threshold


def predict(threshold=0.4):    # 如果单独调用 predict()，使用默认 0.4
    model_path = "stroke_model.cbm"  
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在。")
    catboost_regressor = CatBoostRegressor()
    catboost_regressor.load_model(model_path)
    # # 多视角
    # test_data = pd.concat([
    #     load_data([os.path.join(TRAIN_DIR, dirname) for dirname in ["20241121_184001"]], "left"),
    #     load_data([os.path.join(TRAIN_DIR, dirname) for dirname in ["20241121_184001"]], "right"),
    # ]).sample(frac=1).reset_index(drop=True)
    # 单视角代码（修改为使用所有测试目录）
    test_dirs = [os.path.join(TEST_DIR, d) for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d)) and d.startswith("match")]
    test_data = load_data(test_dirs, single_view=True, shuffle=False)  # 预测时不shuffle，保持时序
    test_data["pred"] = catboost_regressor.predict(test_data[get_feature_cols(PREV_WINDOW_NUM, AFTER_WINDOW_NUM)])
    test_data[["timestamp", "pred", "event_cls", "x", "y", "source_video"]].to_csv("predict.csv", index=False, encoding='utf-8')
    
    # 保存预测的落点数据（pred > threshold的点）
    predicted_bounces = test_data[test_data["pred"] > threshold][["timestamp", "x", "y", "pred", "source_video"]]
    predicted_bounces.to_csv("predicted_bounces.csv", index=False, encoding='utf-8')
    print(f"保存了 {len(predicted_bounces)} 个预测落点到 predicted_bounces.csv (threshold={threshold})")


if __name__ == "__main__":
    best_threshold = main()
    predict(best_threshold)  # 训练模型并使用最佳阈值进行预测

