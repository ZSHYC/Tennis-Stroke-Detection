# 网球击球检测系统 (Tennis Stroke Detection System)



本项目旨在自动识别网球比赛中的击球点（stroke points）。系统接收网球的轨迹数据（包括帧号、时间戳、坐标等），通过时序特征工程和 CatBoost 模型，精准预测每个时间点是否为击球瞬间。



---

## 📁 项目结构

```
classification/
│
├── stroke_model.py              # 特征工程与通用逻辑库
├── train.py                     # 模型训练脚本（核心）
├── predict.py                   # 预测脚本（核心）
├── README.md                    # 项目文档（本文件）
│
├── Tennis-Stroke-Analysis-Data/ # 数据目录
│   ├── 1_parse_logs.py         # 数据预处理脚本
│   ├── 2_prepare_data.py       # 数据准备脚本
│   ├── 3_trajectory_labeler.py # 轨迹标注脚本
│   ├── output/
│   │   ├── parsed_a.csv        # 解析后的原始数据
│   │   └── training_segment.csv # 训练数据（主要输入）
│   └── README/                  # 数据处理文档
│       ├── 1_parse_logs.md
│       ├── 2_prepare_data.md
│       └── 3_trajectory_labeler.md
│
├── stroke_model.cbm             # 训练好的模型文件（输出）
├── best_threshold.txt           # 最佳阈值文件（输出）
├── pr_curve.png                 # PR 曲线图（输出）
│
├── predict.csv                  # 完整预测结果（输出）
├── predict_bounces.csv          # 击球点筛选结果（输出）
│
├── catboost_info/               # CatBoost 训练日志（自动生成）
└── __pycache__/                 # Python 缓存（自动生成）
```
---

## 模型架构

```
输入: 轨迹数据 (CSV)
  ↓
数据预处理 (load_data)
  ├─ 过滤未核对数据
  ├─ 解析击球标签
  └─ 按轨迹分组排序
  ↓
特征工程 (to_features)
  ├─ 前向差分特征 (x_diff_i, y_diff_i)
  ├─ 后向差分特征 (x_diff_inv_i, y_diff_inv_i)
  └─ 比例特征 (x_div_i, y_div_i)
  ↓
数据分割 (80/20, Group Split by traj_id)
  ↓
CatBoost 训练 (3000 iterations, depth=3)
  ├─ 损失函数: RMSE
  ├─ 样本权重: 40:1
  └─ 早停: 100 rounds
  ↓
阈值优化 (evaluate)
  ├─ 遍历阈值: 0.01~0.99 (步长0.01)
  ├─ 优化目标: F-beta (β²=4, 重视召回率)
  └─ 输出: best_threshold.txt
  ↓
模型保存: stroke_model.cbm
  ↓
预测输出
  ├─ predict.csv (完整预测)
  └─ predict_bounces.csv (击球点)
```

---

## 🔧 环境配置


### 

```bash
# 创建 conda 环境
conda create -n tennis python=3.10
conda activate tennis

# 安装依赖
conda install pandas numpy scikit-learn matplotlib
pip install catboost
```

### 验证安装

```python
python -c "import pandas, numpy, catboost, sklearn, matplotlib; print('All dependencies installed successfully!')"
```

---

## 🚀 快速开始

### 步骤 1：准备数据

确保数据文件存在：
```
Tennis-Stroke-Analysis-Data/output/training_segment.csv
```

数据格式要求：
| 列名 | 类型 | 说明 |
|------|------|------|
| `frame_index` | int | 帧号（必需） |
| `x` | float | X 坐标（必需） |
| `y` | float | Y 坐标（必需） |
| `traj_id` | int | 轨迹 ID（必需） |
| `hit_frames_global` | str | 击球帧列表，格式如 "1234,5678" 或 "-1"（必需） |
| `is_checked` | int | 是否已核对，0 或 1（可选） |

### 步骤 2：训练模型

```bash
python train.py
```

**输出文件**：
- `stroke_model.cbm` - 训练好的模型文件
- `best_threshold.txt` - 最佳预测阈值
- `pr_curve.png` - Precision-Recall 曲线图
- `catboost_info/` - CatBoost 训练日志

**预期输出示例**：
```
Loading data from Tennis-Stroke-Analysis-Data/output/training_segment.csv
Filtered unchecked data: 5000 -> 4800
Processing features by trajectory group...
Total data shape: (4500, 25), positive samples: 120
Splitting data: 80 trajectories for training, 20 trajectories for testing.
Train set size: 3600, Test set size: 900

Training...
[CatBoost训练日志]

Best threshold: 0.55 with F-beta: 0.845 (F1: 0.723)
roc 0.892
AUC-PR: 0.678
PR curve saved as pr_curve.png
Best threshold saved to best_threshold.txt

============================================================
训练完成！模型已保存到 stroke_model.cbm
最佳阈值已保存到 best_threshold.txt: 0.5500

要进行预测，请运行: python predict.py
============================================================
```

### 步骤 3：预测击球点

#### 使用最佳阈值（推荐）
```bash
python predict.py
```

#### 使用自定义阈值
```bash
python predict.py --threshold 0.4
```

#### 预测其他数据文件
```bash
python predict.py --data path/to/new_data.csv --threshold 0.5 --output result
```

**输出文件**：
- `predict.csv` - 完整预测结果（所有帧的预测概率）
- `predict_bounces.csv` - 筛选后的击球点（pred > threshold）

---

## 📖 详细使用说明

### 训练脚本 (`train.py`)

#### 核心参数配置

**数据路径配置**（在 `train.py` 中）：
```python
DATA_FILE = "Tennis-Stroke-Analysis-Data/output/training_segment.csv"  # 数据文件路径
```

**特征窗口配置**（在 `stroke_model.py` 中）：
```python
PREV_WINDOW_NUM = 2   # 前向窗口大小（使用前2帧计算差分）
AFTER_WINDOW_NUM = 2  # 后向窗口大小（使用后2帧计算差分）
```

#### 样本权重调整

在 `stroke_model.py` 的 `load_data()` 函数中：
```python
resdf = __add_weight(resdf, {1: 40, 0: 1})  # 正样本权重40，负样本权重1
```

根据数据不平衡程度调整权重，建议范围 20-100。

#### F-beta 参数调整

在 `evaluate()` 函数中：
```python
beta_squared = 4  # 召回率权重是精确率的4倍
```

- `beta_squared = 1`：F1 分数（召回率和精确率等权重）
- `beta_squared = 4`：更重视召回率（宁可误报，不要漏检）
- `beta_squared = 0.5`：更重视精确率（减少误报）

#### CatBoost 参数调整

在 `train()` 函数中：
```python
CatBoostRegressor(
    iterations=3000,        # 迭代次数
    depth=3,                # 树深度（3-10，越大越容易过拟合）
    learning_rate=0.1,      # 学习率（0.01-0.3）
    loss_function='RMSE'    # 损失函数
)
```

---

### 预测脚本 (`predict.py`)

#### 命令行参数详解

```bash
python predict.py [OPTIONS]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data` | str | training_segment.csv | 输入数据文件路径 |
| `--model` | str | stroke_model.cbm | 模型文件路径 |
| `--threshold` | float | None | 自定义阈值（0-1），不指定则使用最佳阈值 |
| `--use-best` | flag | False | 明确使用最佳阈值（从 best_threshold.txt 读取） |
| `--output` | str | predict | 输出文件前缀 |

#### 使用示例

**示例 1：基础预测（使用最佳阈值）**
```bash
python predict.py
```

**示例 2：探索不同阈值的影响**
```bash
# 低阈值（高召回率，多误报）
python predict.py --threshold 0.3 --output pred_low

# 中等阈值
python predict.py --threshold 0.5 --output pred_mid

# 高阈值（高精确率，可能漏检）
python predict.py --threshold 0.7 --output pred_high
```

**示例 3：批量预测多个文件**
```bash
python predict.py --data test_set_1.csv --output test1
python predict.py --data test_set_2.csv --output test2
```

**示例 4：使用不同模型**
```bash
python predict.py --model my_custom_model.cbm --threshold 0.45
```



---

## 🧠 模型原理

### 1. 特征工程原理

#### 前向差分特征 (Backward Difference)
```python
x_diff_1 = x[t-1] - x[t]  # 当前帧与前1帧的X坐标差
y_diff_1 = y[t-1] - y[t]  # 当前帧与前1帧的Y坐标差
```

**物理意义**：捕捉球的运动轨迹变化。击球瞬间，球速和方向会突变。

#### 后向差分特征 (Forward Difference)
```python
x_diff_inv_1 = x[t+1] - x[t]  # 当前帧与后1帧的X坐标差
y_diff_inv_1 = y[t+1] - y[t]  # 当前帧与后1帧的Y坐标差
```

**物理意义**：利用"未来信息"（训练时可用）捕捉击球后的轨迹变化。

#### 比例特征 (Ratio Features)
```python
x_div_1 = x_diff_1 / (x_diff_inv_1 + ε)  # 前后差分的比例
y_div_1 = y_diff_1 / (y_diff_inv_1 + ε)
```

**物理意义**：归一化特征，表示运动趋势的转折程度。击球点处比例通常异常。

### 2. 为什么使用回归而非分类？

虽然最终任务是分类（击球/非击球），但我们使用 **CatBoostRegressor**：

**优势**：
- ✅ 输出连续概率，便于阈值调优
- ✅ 更好地处理不平衡数据（通过样本权重）
- ✅ 平滑的预测曲线，减少抖动



### 3. Group Split 防止数据泄露

**错误做法**（随机分割）：
```python
train_data, test_data = train_test_split(all_data, test_size=0.2)  # ❌ 错误！
```

**问题**：同一条轨迹的前后帧被分到训练集和测试集，模型会"作弊"。

**正确做法**（轨迹分组）：
```python
unique_traj_ids = all_data['traj_id'].unique()
train_ids, test_ids = split_traj_ids(unique_traj_ids, ratio=0.8)
train_data = all_data[all_data['traj_id'].isin(train_ids)]
test_data = all_data[all_data['traj_id'].isin(test_ids)]
```

**保证**：测试集的轨迹完全独立，真实评估泛化能力。

### 4. F-beta 与阈值选择

**标准 F1 分数**：
$$
F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**F-beta 分数（本项目使用 β²=4）**：
$$
F_\beta = (1 + \beta^2) \times \frac{\text{Precision} \times \text{Recall}}{\beta^2 \times \text{Precision} + \text{Recall}}
$$

**权重对比**：
- β=1：Precision 和 Recall 等权重
- β=2：Recall 权重是 Precision 的 4 倍
- β=√4=2：Recall 权重是 Precision 的 **4 倍**（本项目）

**选择理由**：网球击球检测中，**漏检（False Negative）比误报（False Positive）更严重**，因此优先提高召回率。

---

## ⚡ 性能优化

### 1. 训练速度优化

#### 减少迭代次数（快速实验）
```python
CatBoostRegressor(iterations=1000, ...)  # 从 3000 降到 1000
```

#### 增加学习率（需谨慎）
```python
CatBoostRegressor(learning_rate=0.2, ...)  # 从 0.1 提高到 0.2
```

#### 使用 GPU 加速
```python
CatBoostRegressor(..., task_type='GPU', devices='0')
```

### 2. 预测速度优化

#### 批量预测（避免逐帧加载）
```python
# 一次性加载所有数据
all_data = load_data(file_path, shuffle=False)
predictions = model.predict(all_data[features])
```

#### 减少阈值搜索范围（训练时）
```python
# 从 98 个点减少到 18 个点
for threshold in np.arange(0.1, 1.0, 0.05):  # 步长从 0.01 改为 0.05
```

### 3. 内存优化

#### 使用数据类型优化
```python
df['x'] = df['x'].astype('float32')  # 从 float64 降到 float32
df['frame_index'] = df['frame_index'].astype('int32')
```

---

## ❓ 常见问题

### Q1: 训练时报错 "没有足够的数据生成特征"

**原因**：轨迹太短，无法计算窗口特征。

**解决方案**：
```python
# 降低窗口大小
PREV_WINDOW_NUM = 1  # 从 2 改为 1
AFTER_WINDOW_NUM = 1
```

或者检查数据：
```python
df.groupby('traj_id').size().describe()  # 查看轨迹长度分布
```

---

### Q2: 预测时报错 "KeyError: 'timestamp'"

**原因**：CSV 数据中没有 `timestamp` 列。

**解决方案**：`predict.py` 已自动处理，会用 `frame_index` 替代。如果仍报错，手动添加：
```python
df['timestamp'] = df['frame_index']
```

---

### Q3: 模型召回率低，漏检严重

**解决方案**：
1. **降低阈值**：
   ```bash
   python predict.py --threshold 0.3  # 从 0.5 降到 0.3
   ```

2. **增加 beta 值**：
   ```python
   beta_squared = 9  # 从 4 改为 9（召回率权重9倍）
   ```

3. **增加正样本权重**：
   ```python
   resdf = __add_weight(resdf, {1: 100, 0: 1})  # 从 40 提高到 100
   ```

---

### Q4: 模型精确率低，误报严重

**解决方案**：
1. **提高阈值**：
   ```bash
   python predict.py --threshold 0.7  # 从 0.5 提高到 0.7
   ```

2. **降低 beta 值**：
   ```python
   beta_squared = 1  # 改为 F1（等权重）
   ```

3. **减少正样本权重**：
   ```python
   resdf = __add_weight(resdf, {1: 20, 0: 1})  # 从 40 降到 20
   ```

---

### Q5: 训练时间太长

**解决方案**：
1. **减少迭代次数**：
   ```python
   CatBoostRegressor(iterations=1000, ...)
   ```

2. **增加早停轮数**：
   ```python
   early_stopping_rounds=50  # 从 100 改为 50
   ```

3. **使用 GPU**（如果有）：
   ```python
   CatBoostRegressor(..., task_type='GPU')
   ```

---

### Q6: 如何评估模型在新数据上的表现？

**步骤**：
1. 用 `predict.py` 预测新数据
2. 如果有真实标签，手动计算指标：

```python
import pandas as pd
from sklearn.metrics import classification_report

# 加载预测结果
pred = pd.read_csv('predict.csv')

# 计算指标（假设有 event_cls 列）
y_true = pred['event_cls']
y_pred = (pred['pred'] > 0.5).astype(int)

print(classification_report(y_true, y_pred))
```

---

### Q7: 可以用于实时预测吗？

**回答**：理论上可以，但需要注意：

**挑战**：
- 后向特征 (`x_diff_inv_i`) 需要"未来帧"，实时预测无法获取
- 需要缓冲未来 2 帧才能预测当前帧

**解决方案**：
1. 只使用前向特征（修改 `get_feature_cols`）
2. 接受 2 帧延迟（缓冲后预测）
3. 使用滑动窗口实时计算特征
