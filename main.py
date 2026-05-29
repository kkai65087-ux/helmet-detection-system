# -*- coding: utf-8 -*-
"""
实时头盔佩戴检测系统
YOLOv5 目标检测 + OpenCV 人脸识别 + Tkinter GUI
"""
import os
import sys

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
import warnings
warnings.filterwarnings('ignore')

# ========== 配置区 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "weights", "best.pt")
FONT_PATH = "C:/Windows/Fonts/simkai.ttf"
EXCEL_FILE = "results.xlsx"
CONF_THRESHOLD = 0.4        # YOLOv5 置信度阈值
FACE_DB_CACHE = os.path.join(BASE_DIR, "face_db.yml")
FACE_NAMES_CACHE = os.path.join(BASE_DIR, "face_names.pkl")
# ============================

# --- 字体 ---
if os.path.exists(FONT_PATH):
    chinese_font = ImageFont.truetype(FONT_PATH, size=20)
else:
    chinese_font = ImageFont.load_default()

# --- 人脸检测器(OpenCV Haar Cascade, 内置无需下载) ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 全局变量 ---
cap = None
label_img = None
is_camera_running = False
model = None
face_recognizer = None
known_names = []
last_name = "N/A"
last_helmet_status = "N/A"


def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        messagebox.showwarning("警告", f"模型文件未找到: {MODEL_PATH}")
        return False
    model = YOLOv5(MODEL_PATH, device='cpu')
    return True


def build_face_database():
    """
    加载 known_faces/ 照片，Haar Cascade 检测人脸，训练 LBPH，缓存
    """
    global face_recognizer, known_names

    known_dir = os.path.join(BASE_DIR, "known_faces")
    if not os.path.exists(known_dir):
        return False

    # 检查缓存
    names_path = os.path.join(BASE_DIR, "face_names.pkl")
    yml_path = os.path.join(BASE_DIR, "face_db.yml")
    if os.path.exists(yml_path) and os.path.exists(names_path):
        try:
            cache_mtime = os.path.getmtime(yml_path)
            dir_mtime = os.path.getmtime(known_dir)
            if cache_mtime >= dir_mtime:
                face_recognizer = cv2.face.LBPHFaceRecognizer_create()
                face_recognizer.read(yml_path)
                with open(names_path, 'rb') as f:
                    known_names = pickle.load(f)
                print(f"从缓存加载: {known_names}")
                return True
        except Exception:
            pass

    faces_data = []
    labels = []
    label_map = {}

    for filename in os.listdir(known_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            continue
        image_path = os.path.join(known_dir, filename)
        # 用 numpy 读取避免中文路径编码问题
        try:
            img_data = np.fromfile(image_path, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        except Exception:
            continue
        if img is None:
            continue
        name = os.path.splitext(filename)[0]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        if len(faces) == 0:
            print(f"警告: {filename} 未检测到人脸")
            continue

        # 取最大的人脸
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        face_roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100))

        if name not in label_map:
            label_map[name] = len(label_map)
        faces_data.append(face_roi)
        labels.append(label_map[name])
        print(f"已加载: {name}")

    if len(faces_data) == 0:
        return False

    face_recognizer = cv2.face.LBPHFaceRecognizer_create(threshold=80.0)
    face_recognizer.train(faces_data, np.array(labels))
    known_names = list(label_map.keys())

    try:
        face_recognizer.save(os.path.join(BASE_DIR, "face_db.yml"))
        with open(os.path.join(BASE_DIR, "face_names.pkl"), 'wb') as f:
            pickle.dump(known_names, f)
    except Exception:
        pass

    print(f"人脸库就绪: {known_names}")
    return True


def recognize_faces(frame_bgr):
    """Haar Cascade 检测 + LBPH 识别"""
    global face_recognizer, known_names

    if face_recognizer is None:
        return []

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces) == 0:
        return []

    results = []
    for (x, y, w, h) in faces:
        face_roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
        name = "Unknown"
        try:
            label, confidence = face_recognizer.predict(face_roi)
            if confidence < 80:
                name = known_names[label]
        except Exception:
            pass
        results.append((x, y, x + w, y + h, name))

    return results


def start_camera():
    global cap, is_camera_running
    if model is None and not load_model():
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
    global cap, is_camera_running
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
            bbox = draw.textbbox((0, 0), name, font=chinese_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = 60, 24
        if tw > 0 and th > 0:
            draw.rectangle([(left, top - th - 6), (left + tw + 4, top)], fill=(255, 255, 255))
        draw.text((left + 2, top - th - 4), name, font=chinese_font, fill=(0, 0, 0))

    if len(faces) > 0:
        named = [f[4] for f in faces if f[4] != "Unknown"]
        person_name = named[0] if named else f"检测到{len(faces)}人"

    # ====== 时间戳 ======
    current_time = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    draw.text((10, 10), current_time, font=chinese_font, fill=(255, 255, 0))

    # ====== GUI 更新 ======
    imgtk = ImageTk.PhotoImage(image=pil_image)
    label_img.imgtk = imgtk
    label_img.configure(image=imgtk)
    label_img.after(10, update_frame)

    last_name = person_name
    last_helmet_status = helmet_status

    text_box.insert(tk.END, f"时间: {current_time}\n人脸: {person_name}\n头盔: {helmet_status}\n\n")
    text_box.yview(tk.END)

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
            if float(conf) > CONF_THRESHOLD:
                class_name = model.model.names[int(cls)]
                color = (0, 255, 0) if class_name == "helmet" else (0, 0, 255)
                cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])), color, 2)
                cv2.putText(img, f'{class_name} {float(conf):.2f}',
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
