# -*- coding: utf-8 -*-
"""
实时头盔佩戴检测系统
YOLOv5 目标检测 + face_recognition 人脸识别 + Tkinter GUI
检测结果自动导出至 Excel
"""
import os
# 修复 dlib CUDA 驱动兼容性问题
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# 修复 PyTorch 2.6+ 加载旧版 YOLOv5 模型的兼容性问题
import torch
import torch.serialization
_original_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_load

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import scrolledtext
import pandas as pd
from PIL import Image, ImageTk
from PIL import ImageDraw, ImageFont
import numpy as np
import time
import pickle

import face_recognition

# ========== 配置区 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "weights", "best.pt")
FONT_PATH = "C:/Windows/Fonts/simkai.ttf"
EXCEL_FILE = "results.xlsx"
CONF_THRESHOLD = 0.4          # YOLOv5 置信度阈值
FACE_DISTANCE_THRESHOLD = 0.5 # 人脸识别距离阈值 (越小越严格)
# ============================

# --- 字体 ---
if os.path.exists(FONT_PATH):
    chinese_font = ImageFont.truetype(FONT_PATH, size=20)
else:
    chinese_font = ImageFont.load_default()

# --- 全局变量 ---
cap = None
label_img = None
is_camera_running = False
model = None
known_face_encodings = []
known_names = []
face_db_loaded = False
last_name = "N/A"
last_helmet_status = "N/A"


def load_model():
    """加载 YOLOv5 模型"""
    global model
    if not os.path.exists(MODEL_PATH):
        messagebox.showwarning("警告", f"模型文件未找到: {MODEL_PATH}")
        return False
    model = YOLOv5(MODEL_PATH, device='cpu')
    return True


def build_face_database():
    """
    从 known_faces/ 目录加载照片，提取人脸特征并缓存
    """
    global known_face_encodings, known_names, face_db_loaded

    known_dir = os.path.join(BASE_DIR, "known_faces")
    if not os.path.exists(known_dir):
        return False

    cache_path = os.path.join(BASE_DIR, "face_cache.pkl")

    # 尝试加载缓存
    if os.path.exists(cache_path):
        cache_time = os.path.getmtime(cache_path)
        dir_time = os.path.getmtime(known_dir)
        if cache_time >= dir_time:
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                known_face_encodings = data['encodings']
                known_names = data['names']
                face_db_loaded = True
                print(f"从缓存加载了 {len(known_names)} 个人脸: {known_names}")
                return True
            except Exception:
                pass

    # 重新提取特征
    encodings_list = []
    names_list = []

    for filename in os.listdir(known_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            continue
        image_path = os.path.join(known_dir, filename)
        name = os.path.splitext(filename)[0]

        img = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) == 0:
            print(f"警告: {filename} 中未检测到人脸, 跳过")
            continue

        # 取第一张人脸
        encodings_list.append(encodings[0])
        names_list.append(name)
        print(f"已加载: {name}")

    if len(encodings_list) == 0:
        return False

    known_face_encodings = encodings_list
    known_names = names_list
    face_db_loaded = True

    # 缓存
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump({'encodings': known_face_encodings, 'names': known_names}, f)
    except Exception:
        pass

    print(f"人脸库就绪: {known_names}")
    return True


def recognize_faces(frame_bgr):
    """
    检测并识别画面中的人脸
    返回: [(left, top, right, bottom, name), ...]
    """
    rgb = frame_bgr[:, :, ::-1]
    locations = face_recognition.face_locations(rgb)
    if len(locations) == 0:
        return []

    encodings = face_recognition.face_encodings(rgb, locations)
    results = []
    for (top, right, bottom, left), encoding in zip(locations, encodings):
        name = "Unknown"
        if face_db_loaded and len(known_face_encodings) > 0:
            distances = face_recognition.face_distance(known_face_encodings, encoding)
            best_idx = np.argmin(distances)
            if distances[best_idx] < FACE_DISTANCE_THRESHOLD:
                name = known_names[best_idx]
        results.append((left, top, right, bottom, name))
    return results


def start_camera():
    global cap, label_img, is_camera_running
    if model is None:
        if not load_model():
            return
    if cap is None:
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("错误", "无法打开摄像头")
        return
    is_camera_running = True
    update_frame()


def stop_camera():
    global cap, is_camera_running
    if cap:
        cap.release()
        cap = None
    is_camera_running = False
    messagebox.showinfo("提示", "摄像头已关闭")


def update_frame():
    global cap, label_img, is_camera_running
    global last_name, last_helmet_status

    if not is_camera_running:
        return
    ret, frame = cap.read()
    if not ret:
        cap.release()
        cap = None
        messagebox.showwarning("警告", "摄像头连接丢失")
        is_camera_running = False
        return

    helmet_status = "未检测到"
    person_name = "N/A"

    # ====== YOLOv5 头盔检测 ======
    results = model.predict(frame)
    detections = results.xyxy[0]
    best_head_conf = 0.0
    best_helmet_conf = 0.0

    if detections is not None and len(detections) > 0:
        for *xyxy, conf, cls in detections:
            conf_val = float(conf)
            class_name = model.model.names[int(cls)]

            if class_name == "helmet":
                best_helmet_conf = max(best_helmet_conf, conf_val)
            elif class_name == "head":
                best_head_conf = max(best_head_conf, conf_val)

            if conf_val > CONF_THRESHOLD:
                color = (0, 255, 0) if class_name == "helmet" else (0, 0, 255)
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])), color, 2)
                cv2.putText(frame, f'{class_name} {conf_val:.2f}',
                            (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 取最高置信度的类别作为最终判断
    if best_head_conf > CONF_THRESHOLD and best_head_conf >= best_helmet_conf:
        helmet_status = "未戴头盔"
    elif best_helmet_conf > CONF_THRESHOLD and best_helmet_conf > best_head_conf:
        helmet_status = "已戴头盔"

    # ====== 人脸识别 ======
    rgb_frame = frame[:, :, ::-1]
    pil_image = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(pil_image)

    faces = recognize_faces(frame)
    for (left, top, right, bottom, name) in faces:
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)
        try:
            text_bbox = draw.textbbox((0, 0), name, font=chinese_font)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        except Exception:
            text_w, text_h = 60, 24
        if text_w > 0 and text_h > 0:
            draw.rectangle([(left, top - text_h - 6), (left + text_w + 4, top)],
                           fill=(255, 255, 255))
        draw.text((left + 2, top - text_h - 4), name, font=chinese_font, fill=(0, 0, 0))

    if len(faces) > 0:
        named = [f[4] for f in faces if f[4] != "Unknown"]
        person_name = named[0] if named else f"检测到{len(faces)}人"

    # ====== 时间戳 ======
    current_time = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    draw.text((10, 10), current_time, font=chinese_font, fill=(255, 255, 0))

    # ====== 更新 GUI ======
    imgtk = ImageTk.PhotoImage(image=pil_image)
    label_img.imgtk = imgtk
    label_img.configure(image=imgtk)
    label_img.after(10, update_frame)

    last_name = person_name
    last_helmet_status = helmet_status

    # ====== 日志 ======
    text_box.insert(tk.END, f"时间: {current_time}\n人脸: {person_name}\n头盔: {helmet_status}\n\n")
    text_box.yview(tk.END)

    # ====== Excel 导出 ======
    df_new = pd.DataFrame({'时间': [current_time], '姓名': [person_name], '是否戴了头盔': [helmet_status]})
    if os.path.exists(EXCEL_FILE):
        try:
            df = pd.read_excel(EXCEL_FILE)
            df = pd.concat([df, df_new], ignore_index=True)
        except Exception:
            df = df_new
    else:
        df = df_new
    try:
        df.to_excel(EXCEL_FILE, index=False)
    except Exception:
        pass


def upload_file():
    if model is None and not load_model():
        return
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if not file_path:
        return
    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("错误", "无法加载图片")
        return
    results = model.predict(img)
    detections = results.xyxy[0]
    if detections is not None:
        for *xyxy, conf, cls in detections:
            conf_val = float(conf)
            class_name = model.model.names[int(cls)]
            if conf_val > CONF_THRESHOLD:
                color = (0, 255, 0) if class_name == "helmet" else (0, 0, 255)
                cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])), color, 2)
                cv2.putText(img, f'{class_name} {conf_val:.2f}',
                            (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow('检测结果', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def take_photo():
    if cap is None or not cap.isOpened():
        messagebox.showwarning("警告", "请先打开摄像头")
        return
    ret, frame = cap.read()
    if ret:
        save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if save_path:
            cv2.imwrite(save_path, frame)
            messagebox.showinfo("成功", f"照片已保存至 {save_path}")


# ========== GUI ==========
root = tk.Tk()
root.title("头盔佩戴检测系统 - YOLOv5")

label_img = tk.Label(root)
label_img.pack()

btn_frame = tk.Frame(root)
btn_frame.pack(pady=5)

tk.Button(btn_frame, text="打开摄像头", command=start_camera, width=12).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="停止摄像头", command=stop_camera, width=12).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="上传图片检测", command=upload_file, width=12).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="拍照保存", command=take_photo, width=12).pack(side=tk.LEFT, padx=5)

text_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10)
text_box.pack(side=tk.BOTTOM, padx=10, pady=10)


def on_closing():
    global cap
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)

# ========== 启动 ==========
if __name__ == "__main__":
    try:
        from yolov5 import YOLOv5
        load_model()
        build_face_database()
    except Exception as e:
        print(f"初始化失败: {e}")
    root.mainloop()
