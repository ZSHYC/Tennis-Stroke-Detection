import json
import pandas as pd
import re
import os
from tqdm import tqdm

LOG_FILES = {
    "a": "a.log/a.log",
    "b": "b.log/b.log"
}
OUTPUT_DIR = "output"


def parse_single_log(camera_name, file_path):
    print(f"\n 正在启动解析任务: 摄像头 {camera_name}")
    print(f"   文件路径: {file_path}")
    
    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f" 错误：找不到文件！请检查路径是否正确。")
        return

    extracted_data = []
    
    # 2. 逐行读取日志
    # encoding='utf-8' 是为了防止中文乱码，如果报错可以试着改成 'gbk'
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 使用 tqdm 显示进度条，因为日志可能很大，读取需要时间
            lines = f.readlines()
            print(f"   文件读取完毕，共 {len(lines)} 行，开始提取数据...")
            
            for line in tqdm(lines, desc=f"解析 {camera_name}"):
                # 【关键技术点】使用正则表达式 (Regex) 提取 JSON
                # 它的意思是：找到以 { 开头，包含 "frame_index"，并以 } 结尾的一段文字
                match = re.search(r'(\{.*"frame_index".*\})', line)
                
                if match:
                    json_str = match.group(1)
                    try:
                        # 把字符串变成 Python 字典
                        item = json.loads(json_str)
                        
                        # 提取基础信息
                        frame_idx = item.get('frame_index')
                        ts = item.get('timestamp')
                        
                        # 【关键业务逻辑】筛选主球 (is_main: true)
                        # 系统可能同时检测到多个物体，我们只关心被标记为"主球"的那一个
                        positions = item.get('positions', [])
                        for pos in positions:
                            if pos.get('is_main', False) == True:
                                # 提取成功！存入列表
                                extracted_data.append({
                                    'frame_index': frame_idx,
                                    'timestamp': ts,
                                    'traj_id': pos['trajectory_id'],
                                    'x': pos['x'],
                                    'y': pos['y']
                                })
                                break # 一帧只取一个主球，找到就溜
                                
                    except json.JSONDecodeError:
                        # 如果某一行 JSON 格式坏了，直接跳过，不让程序崩溃
                        continue
    except Exception as e:
        print(f" 读取文件发生未知错误: {e}")
        return

    # 3. 数据保存
    if len(extracted_data) > 0:
        df = pd.DataFrame(extracted_data)
        
        # 【极其重要】按帧号排序
        # 日志是多线程写入的，可能会乱序。不排序的话，轨迹会乱跳.
        df = df.sort_values(by='frame_index').reset_index(drop=True)
        
        # 创建输出文件夹
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, f"parsed_{camera_name}.csv")
        
        df.to_csv(save_path, index=False)
        print(f"✅ {camera_name} 机位解析成功！")
        print(f"   - 有效数据点: {len(df)}")
        print(f"   - 轨迹 ID 数量: {df['traj_id'].nunique()} 个")
        print(f"   - 结果已保存至: {save_path}")
    else:
        print(f" 警告: {camera_name} 日志里没有提取到任何有效数据！请检查日志内容格式。")

if __name__ == "__main__":
    # 循环处理 A 和 B 两个文件
    for name, path in LOG_FILES.items():
        parse_single_log(name, path)
