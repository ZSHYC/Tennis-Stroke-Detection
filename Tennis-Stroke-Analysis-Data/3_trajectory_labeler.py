import cv2
import pandas as pd
import numpy as np
import os
import datetime


CSV_PATH = "output/training_segment.csv"
BG_PATH = "left.jpg" 
LOG_FILE = "labeling_history.log"

FORCE_START_INDEX = 282 # -1自动, 0强制从头

COORD_W, COORD_H = 1280, 720
DISPLAY_W = 1080 
DISPLAY_H = 608 

SCALE_X = DISPLAY_W / COORD_W
SCALE_Y = DISPLAY_H / COORD_H

def log_action(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    print(full_msg)
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(full_msg + "\n")

def draw_text_with_outline(img, text, pos, scale, color, thickness):
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, (0,0,0), thickness+3)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness)

def play_trajectory(group, bg_img, traj_id, progress_str, current_index, total_traj):
    coords = group[['x', 'y']].values 
    coords[:, 0] *= SCALE_X
    coords[:, 1] *= SCALE_Y
    frames = group['frame_index'].values
    
    # 读取已有标记
    raw_hits = str(group.iloc[0]['hit_frames_global'])
    current_hit_indices = [] 
    if raw_hits and raw_hits.lower() not in ['nan', 'none', '', '-1']:
        try:
            parts = raw_hits.split(',')
            for p in parts:
                val = float(p)
                if int(val) != -1:
                    matches = np.where(frames == int(val))[0]
                    if len(matches) > 0: current_hit_indices.append(matches[0])
        except: pass
            
    i = 0
    paused = False 
    
    # 自动定位到第一个标记点
    if len(current_hit_indices) > 0:
        i = current_hit_indices[0]
        paused = True
    
    while True:
        img = bg_img.copy()
        
        # 1. 绘制历史
        for j in range(i + 1):
            px, py = int(coords[j][0]), int(coords[j][1])
            if px > 1: cv2.circle(img, (px, py), 3, (150, 150, 150), -1)
            
        # 2. 绘制当前
        cur_x, cur_y = int(coords[i][0]), int(coords[i][1])
        ball_color = (0, 0, 255) if paused else (0, 255, 0)
        if cur_x > 1:
            cv2.circle(img, (cur_x, cur_y), 12, ball_color, -1)
            cv2.circle(img, (cur_x, cur_y), 12, (255, 255, 255), 2)
            
        # 3. 绘制所有标记
        is_hit_global = len(current_hit_indices) > 0
        
        for h_idx in current_hit_indices:
            hx, hy = int(coords[h_idx][0]), int(coords[h_idx][1])
            if hx > 1:
                cv2.circle(img, (hx, hy), 30, (0, 255, 255), 3)
                if i == h_idx: 
                    draw_text_with_outline(img, "HIT MARKED", (hx+40, hy), 0.8, (0, 255, 255), 2)

        # 4. UI
        top_bar = np.zeros((80, DISPLAY_W, 3), dtype=np.uint8)
        bottom_bar = np.zeros((60, DISPLAY_W, 3), dtype=np.uint8)
        
        status_txt = "PAUSED (Use , .)" if paused else "PLAYING"
        status_col = (0, 0, 255) if paused else (0, 255, 0)
        
        hit_status_txt = "[HAS HIT]" if is_hit_global else "[NO HIT]"
        hit_status_col = (0, 255, 0) if is_hit_global else (150, 150, 150)
        
        draw_text_with_outline(top_bar, f"{progress_str} | ID: {traj_id}", (20, 35), 0.7, (255, 255, 255), 1)
        draw_text_with_outline(top_bar, hit_status_txt, (400, 35), 0.8, hit_status_col, 2)
        
        hits_count = len(current_hit_indices)
        draw_text_with_outline(top_bar, f"Hits: {hits_count}", (20, 70), 0.9, (0, 255, 255) if hits_count>0 else (200,200,200), 2)
        draw_text_with_outline(top_bar, f"Mode: {status_txt}", (300, 70), 0.7, status_col, 1)
        
        # 底部帧号
        current_frame_real = frames[i]
        frame_status = f"Frame: {current_frame_real}"
        if i in current_hit_indices:
            frame_status += " (HIT!)"
            frame_col = (0, 255, 255)
        else:
            frame_col = (255, 255, 255)
        draw_text_with_outline(img, frame_status, (50, 50), 1.0, frame_col, 2)
        
        draw_text_with_outline(bottom_bar, "[1]:Toggle Hit | [0]:Clear All | [Enter]:Next | [Q]:Quit", (20, 40), 0.6, (255, 255, 255), 1)

        final_view = np.vstack((top_bar, img, bottom_bar))
        cv2.imshow("Trajectory Labeler v6.5", final_view)
        
        wait_t = 0 if paused else 30
        key = cv2.waitKey(wait_t)
        
        # --- 导航 ---
        if key == ord('q'): 
            log_action("用户选择退出")
            return "quit", None
        if key == ord('a'): 
            log_action(f"ID {traj_id}: 返回上一条")
            return "prev_traj", None 
        if key == ord('d') or key == 13: 
            final_global_hits = [frames[idx] for idx in current_hit_indices]
            if final_global_hits:
                log_action(f"ID {traj_id}: 确认击球，共 {len(final_global_hits)} 处: {final_global_hits}")
            else:
                log_action(f"ID {traj_id}: 确认无击球")
            return "next_traj", final_global_hits
            
        if key == 32: paused = not paused 
        
        # --- 编辑 ---
        if paused:
            if key == ord(','): i = max(0, i - 1)
            if key == ord('.'): i = min(len(coords) - 1, i + 1)
            
            # 【按 1】切换标记
            if key == ord('1'):
                if i in current_hit_indices: 
                    current_hit_indices.remove(i)
                    log_action(f"  ->   [撤销] 移除标记: 帧 {frames[i]}")
                else: 
                    current_hit_indices.append(i)
                    log_action(f"  ->  [添加] 标记击球: 帧 {frames[i]}")
                current_hit_indices.sort()
                
            # 【按 0】一键清空
            if key == ord('0'):
                if len(current_hit_indices) > 0:
                    log_action(f"  ->  [清空] 移除了所有 {len(current_hit_indices)} 个标记")
                    current_hit_indices = []
                else:
                    print("  (当前无标记)")
        else:
            i = (i + 1) % len(coords)

def label_tool():
    if not os.path.exists(CSV_PATH): print("❌ 找不到 CSV"); return
    df = pd.read_csv(CSV_PATH, dtype={'traj_id': int, 'frame_index': int})
    bg_raw = cv2.imread(BG_PATH)
    if bg_raw is None: print("❌ 找不到背景图"); return
    
    bg_resized = cv2.resize(bg_raw, (DISPLAY_W, DISPLAY_H))
    
    if 'hit_frames_global' not in df.columns:
        df['hit_frames_global'] = -1
        df['hit_frames_global'] = df['hit_frames_global'].astype(str)
    
    traj_ids = df['traj_id'].unique()
    total_traj = len(traj_ids)
    
    curr = 0
    if FORCE_START_INDEX >= 0:
        curr = FORCE_START_INDEX
        log_action(f" 强制启动，从第 {curr} 条轨迹开始 ({curr+1}/{total_traj})")
    else:
        for i, tid in enumerate(traj_ids):
            mask = df['traj_id'] == tid
            if 'is_checked' not in df.columns: df['is_checked'] = 0
            if df.loc[mask, 'is_checked'].iloc[0] == 0:
                curr = i; break
        log_action(f" 自动启动，从第 {curr} 条轨迹开始 ({curr+1}/{total_traj})")
    
    while 0 <= curr < total_traj:
        tid = traj_ids[curr]
        mask = df['traj_id'] == tid
        group = df[mask]
        
        if len(group) < 5: curr += 1; continue
            
        prog = f"Progress: {curr}/{total_traj-1}"
        log_action(f" 正在处理轨迹 {curr+1}/{total_traj} (ID: {tid})")
        action, hit_globals = play_trajectory(group, bg_resized, tid, prog, curr, total_traj)
        
        if action == "quit":
            df.to_csv(CSV_PATH, index=False)
            break
            
        if action == "prev_traj": 
            curr = max(0, curr - 1)
            
        if action == "next_traj":
            df.loc[mask, 'is_checked'] = 1
            if hit_globals:
                hit_str = ",".join(map(str, hit_globals))
                df.loc[mask, 'hit_frames_global'] = hit_str
                df.loc[mask, 'is_hit'] = 1 
            else:
                df.loc[mask, 'hit_frames_global'] = "-1"
                df.loc[mask, 'is_hit'] = 0
            curr += 1 
            log_action(f"✅ 完成轨迹 {curr}/{total_traj}")
        
        if curr % 5 == 0: df.to_csv(CSV_PATH, index=False)
            
    df.to_csv(CSV_PATH, index=False)
    cv2.destroyAllWindows()
    log_action("✅ 全部完成！")

if __name__ == "__main__":
    label_tool()