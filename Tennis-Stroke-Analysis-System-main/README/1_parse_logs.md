###  `1_parse_logs.py` (Log Parser)

#### **1. 功能概述 (Functionality)**
**数据清洗核心模块**
本脚本专门用于处理原始的非结构化日志文件（`.log`），将其中的杂乱文本转化为结构化、可用的 CSV 数据表。它是整个数据管线的“入口”，没有它，后续的模型训练和推理都无法进行。

#### **2. 输入输出 (I/O)**
*   **输入 (Input)**:
    *   `a.log` / `b.log`: 包含系统运行信息的原始日志文件。每一行都可能包含时间戳、坐标、系统状态等杂乱信息。
*   **输出 (Output)**:
    *   `parsed_a.csv` / `parsed_b.csv`: 清洗后的标准数据表。
    *   **列定义**:
        *   `frame_index`: 绝对帧号 (用于排序)。
        *   `timestamp`: 毫秒级时间戳 (用于找图)。
        *   `traj_id`: 轨迹 ID (用于区分不同的球)。
        *   `x`, `y`: 球的绝对坐标 (Pixel Level)。

#### **3. 核心算法 (Core Logic)**
1.  **正则提取 (Regex Extraction)**:
    *   使用正则表达式 `r'(\{.*"frame_index".*\})'` 从每一行日志中精准抠出 JSON 数据包，忽略掉无关的 `INFO`, `DEBUG` 前缀。
2.  **主球筛选 (Main Ball Filtering)**:
    *   JSON 数据中可能包含多个检测目标（干扰项）。脚本会遍历 `positions` 列表，只保留标记为 `is_main: true` 的目标，确保数据的纯净度。
3.  **时序重组 (Time-Series Sorting)**:
    *   由于日志写入可能是异步的，存在乱序风险。脚本在保存前强制按 `frame_index` 进行升序排列，保证时间轴的物理连续性。

#### **4. 使用指南 (Usage)**
1.  **配置**: 修改 `LOG_FILES` 字典中的路径，指向你的真实 log 文件。
2.  **运行**: 直接执行 `python 1_parse_logs.py`。
3.  **检查**: 运行结束后，请务必检查 `output` 目录下生成的 csv 文件大小，确保不是空的。
