### `2_prepare_data.py` (Data Extractor)

#### **1. 功能概述 (Functionality)**
**数据集截取与重构模块**。
原始日志文件（如 `parsed_a.csv`）包含长达 4 小时的比赛数据，数据量过于庞大，不适合一次性进行人工标注和训练。
本脚本的作用是：
1.  **切片**：从海量数据中，根据用户设定的时长（如 10 分钟、100 分钟），截取一段连续的、有代表性的数据片段。
2.  **重构**：将截取的数据按 `traj_id`（轨迹ID）分组，并进行**插值补全**，修复丢帧问题。
3.  **初始化**：为截取的数据添加打标所需的空列（如 `is_hit`, `hit_frame_global`），生成待标注的 `training_segment.csv`。

#### **2. 输入输出 (I/O)**
*   **输入 (Input)**:
    *   `parsed_a.csv`: 由 `1_parse_logs.py` 生成的全量坐标数据。
*   **输出 (Output)**:
    *   `training_segment.csv`: 截取并清洗后的数据片段，包含了所有打标所需的预留字段。

#### **3. 核心逻辑详解 (Key Logic)**

**A. 截取逻辑 (Time Window Selection)**
*   脚本默认采取 **“取中段”** 策略：
    ```python
    mid_ts = min_ts + (max_ts - min_ts) / 2
    start_ts = mid_ts - (CLIP_DURATION_MS / 2)
    end_ts = mid_ts + (CLIP_DURATION_MS / 2)
    ```
    *   **原因**：比赛开头通常是热身或调试，结尾可能是垃圾时间，中间段的数据质量通常最高。
    *   **自定义**：你也可以修改 `start_ts = min_ts` 来从头开始截取。

**B. 轨迹修复 (Trajectory Interpolation)**
*   **问题**：原始 Log 经常丢帧（例如 Frame 30 -> 35），导致轨迹不连贯。
*   **解决**：`interpolate_trajectory` 函数会：
    1.  找到该轨迹的 `min_frame` 和 `max_frame`。
    2.  生成一个完整的整数索引序列（不缺号）。
    3.  利用 `df.interpolate(method='linear')` 自动填补缺失的 `x, y, timestamp`。
    *   **价值**：这一步对于 LSTM 模型至关重要，因为它依赖于连续的时间步长。

**C. CSV 结构组织 (Schema Design)**
生成的 CSV 每一行代表一个**帧 (Frame Point)**，但标注是基于**轨迹 (Trajectory)** 的。
*   `traj_id`: 轨迹唯一标识符。
*   `frame_index`: 绝对帧号。
*   `is_hit`: **(预留列)** 默认填 `0`。打标时会被修改。
*   `hit_frame_global`: **(预留列)** 默认填 `-1`。打标时会填入具体的击球帧号。
*   `is_checked`: **(预留列)** 默认填 `0`。打标完成后会被改为 `1`。

#### **4. 常见问题 (FAQ)**

**Q1: 如果我想改变训练时长，怎么操作？**
*   **操作**：修改代码顶部的 `CLIP_DURATION_MS` 变量。
    *   例如：`600000` = 10分钟，`6000000` = 100分钟。
*   **注意**：时长越长，生成的 CSV 行数越多，打标的工作量越大。建议初次尝试先用 10-20 分钟。

**Q2: 改变时长后，之前标好的数据还能保留吗？**
*   **不能直接保留**。
    *   如果你重新运行这个脚本，它会**覆盖**掉旧的 `training_segment.csv`。
    *   **正确做法**：如果你想保留旧的标注，请先手动将旧的 CSV 重命名（例如 `training_segment_old.csv`），或者在生成新 CSV 后，用 Pandas 代码将旧 CSV 中的标注列 `merge` 回来（前提是 `traj_id` 和 `frame_index` 能对上）。

**Q3: 为什么生成的 CSV 行数比 Log 里截取的行数多？**
*   这是正常的。因为 `interpolate_trajectory` 会补全丢失的帧，所以行数会增加。这代表你的数据变得更完整了。
