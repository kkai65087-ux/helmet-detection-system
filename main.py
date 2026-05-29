# -*- coding: utf-8 -*-
"""
实时头盔佩戴检测系统
基于 YOLOv5 目标检测 + OpenCV 人脸检测 + Tkinter GUI
检测结果自动导出至 Excel
"""
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
import os
import time
import sys

# ========== 配置区 ==========
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "best.pt")
FONT_PATH = "C:/Windows/Fonts/simkai.ttf"
EXCEL_FILE = "results.xlsx"
# ============================

# --- 字体 ---
if os.path.exists(FONT_PATH):
    chinese_font = ImageFont.truetype(FONT_PATH, size=20)
else:
    chinese_font = ImageFont.load_default()

# --- 尝试加载人脸识别库（可选） ---
face_recognition = None
try:
    import face_recognition as fr
    face_recognition = fr
except ImportError:
    pass

# --- 全局变量 ---
cap = None
label_img = None
is_camera_running = False
model = None
known_face_encodings = []
known_names = []
face_model_loaded = False
# 默认值，防止变量未定义
last_name = "N/A"
last_helmet_status = "N/A"


def load_model():
    """加载 YOLOv5 模型"""
    global model
    if not os.path.exists(MODEL_PATH):
        messagebox.showwarning("警告", f"模型文件未找到: {MODEL_PATH}\n请将 best.pt 放入 weights 文件夹")
        return False
    model = YOLOv5(MODEL_PATH, device='cpu')
    return True


def load_known_faces():
    """加载已知人脸库（需要 face_recognition 库）"""
    global known_face_encodings, known_names, face_model_loaded
    if face_recognition is None:
        return
    known_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_faces")
    if not os.path.exists(known_dir):
        return
    count = 0
    for filename in os.listdir(known_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            image_path = os.path.join(known_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            for encoding in encodings:
                if encoding is not None:
                    name = os.path.splitext(filename)[0]
                    known_face_encodings.append(encoding)
                    known_names.append(name)
                    count += 1
    if count > 0:
        face_model_loaded = True
        print(f"已加载 {count} 个人脸特征")


def recognize_faces_cv2(frame):
    """使用 OpenCV 做简单人脸检测（不识别身份）"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    results = []
    for (x, y, w, h) in faces:
        results.append((y, x + w, y + h, x))  # (top, right, bottom, left) 格式
    return results, ["Person"] * len(results)


def recognize_faces_dlib(rgb_frame):
    """使用 face_recognition 做人脸识别"""
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    names = []
    for encoding in face_encodings:
        if len(known_face_encodings) == 0:
            names.append("Unknown")
        else:
            distances = face_recognition.face_distance(known_face_encodings, encoding)
            best_idx = np.argmin(distances)
            if distances[best_idx] < 0.6:
                names.append(known_names[best_idx])
            else:
                names.append("Unknown")
    return face_locations, names


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

    helmet_status = "N/A"
    person_name = "N/A"

    # --- YOLOv5 头盔检测 ---
    results = model.predict(frame)
    detections = results.xyxy[0]
    if detections is not None and len(detections) > 0:
        for *xyxy, conf, cls in detections:
            class_name = model.model.names[int(cls)]
            label = f'{class_name} {conf:.2f}'
            color = (0, 255, 0) if class_name == "helmet" else (0, 0, 255)
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                          (int(xyxy[2]), int(xyxy[3])), color, 2)
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # 记录最后一个检测结果
            if class_name == "helmet":
                helmet_status = "已戴头盔"
            elif class_name == "head":
                helmet_status = "未戴头盔"

    # --- 人脸检测/识别 ---
    rgb_frame = frame[:, :, ::-1]
    pil_image = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(pil_image)

    if face_recognition is not None and face_model_loaded:
        # 使用 face_recognition（可识别身份）
        face_locations, face_names = recognize_faces_dlib(rgb_frame)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            person_name = name
            if left < right and top < bottom:
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)
                text_bbox = draw.textbbox((0, 0), name, font=chinese_font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                if text_w > 0 and text_h > 0:
                    draw.rectangle([(left + 2, bottom - text_h - 10),
                                    (left + text_w + 4, bottom)], fill=(255, 255, 255))
                draw.text((left + 4, bottom - text_h - 8), name, font=chinese_font, fill=(0, 0, 0))
    else:
        # 使用 OpenCV Haar Cascade（仅检测人脸位置，不识别身份）
        face_locations, face_names = recognize_faces_cv2(frame)
        for (top, right, bottom, left), _ in zip(face_locations, face_names):
            if left < right and top < bottom:
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)
                draw.text((left + 4, top - 20), "Person", font=chinese_font, fill=(0, 255, 255))
        if len(face_names) > 0:
            person_name = f"检测到{len(face_names)}人"

    # --- 绘制时间戳 ---
    current_time = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    frame_h, frame_w = frame.shape[:2]
    timestamp_text = f"{current_time}"
    draw.text((10, 10), timestamp_text, font=chinese_font, fill=(255, 255, 0))

    # --- 更新 Tkinter 画面 ---
    imgtk = ImageTk.PhotoImage(image=pil_image)
    label_img.imgtk = imgtk
    label_img.configure(image=imgtk)
    label_img.after(10, update_frame)

    # --- 记住本次结果 ---
    last_name = person_name
    last_helmet_status = helmet_status

    # --- 更新日志框 ---
    text_display = f"时间: {current_time}\n识别到的人脸: {person_name}\n头盔状态: {helmet_status}\n\n"
    text_box.insert(tk.END, text_display)
    text_box.yview(tk.END)

    # --- 导出到 Excel ---
    record = {'时间': [current_time], '姓名': [person_name], '是否戴了头盔': [helmet_status]}
    new_df = pd.DataFrame(record)
    if os.path.exists(EXCEL_FILE):
        try:
            df = pd.read_excel(EXCEL_FILE)
            df = pd.concat([df, new_df], ignore_index=True)
        except Exception:
            df = new_df
    else:
        df = new_df
    try:
        df.to_excel(EXCEL_FILE, index=False)
    except Exception:
        pass  # 文件被占用时静默跳过


def upload_file():
    """上传图片文件检测"""
    if model is None and not load_model():
        return
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
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
            class_name = model.model.names[int(cls)]
            label = f'{class_name} {conf:.2f}'
            color = (0, 255, 0) if class_name == "helmet" else (0, 0, 255)
            cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])),
                          (int(xyxy[2]), int(xyxy[3])), color, 2)
            cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow('检测结果', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def take_photo():
    """拍照保存"""
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

btn_start = tk.Button(btn_frame, text="打开摄像头", command=start_camera, width=12)
btn_start.pack(side=tk.LEFT, padx=5)

btn_stop = tk.Button(btn_frame, text="停止摄像头", command=stop_camera, width=12)
btn_stop.pack(side=tk.LEFT, padx=5)

btn_upload = tk.Button(btn_frame, text="上传图片检测", command=upload_file, width=12)
btn_upload.pack(side=tk.LEFT, padx=5)

btn_photo = tk.Button(btn_frame, text="拍照保存", command=take_photo, width=12)
btn_photo.pack(side=tk.LEFT, padx=5)

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
    # 尝试加载模型
    try:
        from yolov5 import YOLOv5
        load_model()
        load_known_faces()
    except Exception as e:
        print(f"模型加载失败: {e}")
    root.mainloop()
