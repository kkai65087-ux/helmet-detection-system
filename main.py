# -*- coding: utf-8 -*-
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import scrolledtext
import pandas as pd
from PIL import Image, ImageTk
from PIL import ImageDraw, ImageFont

import face_recognition
import numpy as np
import threading
import os
import time  # 导入time库以获取当前时间
from yolov5 import YOLOv5  # 假设YOLOv5已经被正确安装和配置
import openpyxl
from openpyxl.utils import get_column_letter
import time

chinese_font_size = 24  # 原始大小基础上增加一些值来模拟加粗
chinese_font_path = 'C:/Windows/Fonts/simkai.ttf'  # 替换为您的中文字体文件路径
chinese_font = ImageFont.truetype(chinese_font_path, size=20)  # 调整字体大小

# 全局变量
cap = None
label_img = None
is_camera_running = False
save_path = None  # 用于保存拍照的文件路径
model = YOLOv5("D:/yolov5/yolov5-master/weights/best.pt", device='cpu')  # 加载预训练的YOLOv5模型
known_images_dir = "path_to_known_images_folder"  # 设置包含已知人物脸部图像的文件夹路径11111
known_face_encodings = []
known_names = []
# name2=''



# 加载已知人脸的图像和名字
for filename in os.listdir(known_images_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
        image_path = os.path.join(known_images_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        for encoding in encodings:  # 支持每张图像中有多张人脸
            if encoding is not None:
                name = os.path.splitext(filename)[0]
                known_face_encodings.append(encoding)
                if name not in known_names:
                    known_names.append(name)


def start_camera():
    global cap, label_img, is_camera_running
    if cap is None:
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to open camera")
        return
    is_camera_running = True
    update_frame()


def stop_camera():
    global cap, is_camera_running
    if cap:
        cap.release()
        cap = None
    is_camera_running = False
    messagebox.showinfo("Info", "Camera stopped")


def update_frame():
    global cap, label_img, is_camera_running, chinese_font
    if not is_camera_running:
        return
    ret, frame = cap.read()
    if not ret:
        cap.release()
        cap = None
        messagebox.showwarning("Warning", "Lost connection to camera")
        is_camera_running = False
        return

    # 使用YOLOv5进行目标检测
    yolov5_results = model.predict(frame)
    for *xyxy, conf, cls in yolov5_results.xyxy[0]:
        label = f'{model.model.names[int(cls)]} {conf:.2f}'
        label_name = f'{model.model.names[int(cls)]}'

        if label_name == "helmet":
            name2 = str("戴了头盔")
        elif label_name == "head":
            name2 = str("没戴头盔")

        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 将BGR转换为RGB
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_locations_and_names = {}
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        best_match_distance = face_distances[best_match_index]
        if best_match_distance < 0.6:  # 阈值可以根据实际情况调整
            name = known_names[best_match_index]
        else:
            name = "Unknown"
        face_locations_and_names[(top, left)] = name

    # 创建一个Pillow图像对象
    pil_image = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(pil_image)

    # 在检测到的人脸框上绘制中文名字
    for (top, left), name in face_locations_and_names.items():
        for (t, r, b, l) in face_locations:
            if (t, l) == (top, left):
                # 检查人脸框的有效性
                if l < r and t < b:
                    text_position = (l + 6, b - chinese_font_size - 6)  # 调整位置以适应新的字体大小
                    # 计算背景框大小（考虑文本高度和额外空间）
                    text_bg_width, text_bg_height = draw.textsize(name, font=chinese_font)
                    text_bg_x = l + 5
                    text_bg_y = b - text_bg_height - 10  # 留出一些空间

                    # 打印调试信息
                    # print(f"Drawing text background at ({text_bg_x}, {text_bg_y}) with size ({text_bg_width}, {text_bg_height})")

                    # 绘制背景矩形（添加条件检查以避免错误）
                    if text_bg_width > 0 and text_bg_height > 0:
                        draw.rectangle(
                            [(text_bg_x, text_bg_y), (text_bg_x + text_bg_width, text_bg_y + text_bg_height)],
                            fill=(255, 255, 255))

                    # 绘制文本
                    draw.text(text_position, name, font=chinese_font, fill=(0, 0, 0))
                    break

    # 获取当前时间并格式化为字符串
    current_time = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    frame_height, frame_width = frame.shape[:2]
    # 在Pillow图像上绘制时间戳
    text_position = (frame_width - 200, frame_height - 450)  # 调整位置以适应中文长度
    draw.text(text_position, current_time, font=chinese_font, fill=(255, 255, 0))

    # 将Pillow图像转换为Tkinter可以显示的格式
    imgtk = ImageTk.PhotoImage(image=pil_image)
    label_img.imgtk = imgtk
    label_img.configure(image=imgtk)
    label_img.after(10, update_frame)

    # 格式化并更新文本框内容
    text_to_display = f"时间: {current_time}\n"
    text_to_display += f"识别到的人脸: {name}\n"
    text_to_display += f"头盔状态: {name2}\n\n"  # 添加换行以便分隔每条记录

    # 插入新内容到文本框，并自动滚动到底部
    text_box.insert(tk.END, text_to_display)
    text_box.yview(tk.END)  # 滚动到文本框底部

    data = current_time, name, name2
    print(data[0],data[1],data[2])

    # 列名
    columns = ['时间', '姓名', '是否戴了头盔']

    # 将元组转换为字典
    record = dict(zip(columns, data))

    # Excel 文件名
    excel_file = 'results.xlsx'

    # 检查文件是否存在
    if os.path.exists(excel_file):
        # 读取现有的 Excel 文件
        df = pd.read_excel(excel_file)
        # 将新记录转换为 DataFrame 并追加到现有 DataFrame
        new_df = pd.DataFrame([record])
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        # 如果文件不存在，则创建一个新的 DataFrame
        df = pd.DataFrame([record])

    # 将 DataFrame 保存回 Excel 文件
    df.to_excel(excel_file, index=False)

    print(f"数据已保存到 {excel_file}")
    return current_time, name, name2

def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Failed to load image")
            return
        # 使用YOLOv5进行目标检测
        results = model.predict(img)
        # 在帧上绘制检测结果
        for *xyxy, conf, cls in results.xyxy[0]:
            label = f'{model.model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
            cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        # 显示结果图像
        cv2.imshow('Detected Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def take_photo():
    global save_path
    ret, frame = cap.read()
    if ret:
        # 获取文件名和保存路径
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
        if save_path:
            # 保存图像
            cv2.imwrite(save_path, frame)
            messagebox.showinfo("Success", f"Photo saved to {save_path}")


# 创建主窗口
root = tk.Tk()
root.title("YOLOv5 Real-time Object Detection with GUI")
# 创建Label用于显示视频流
label_img = tk.Label(root)
label_img.pack()
# 创建按钮
btn_start_camera = tk.Button(root, text="打开摄像头", command=start_camera)
btn_start_camera.pack(side=tk.LEFT, padx=10, pady=10)
btn_stop_camera = tk.Button(root, text="停止摄像头", command=stop_camera)
btn_stop_camera.pack(side=tk.LEFT, padx=10, pady=10)
btn_take_photo = tk.Button(root, text="保存图片", command=take_photo)
btn_take_photo.pack(side=tk.RIGHT, padx=10, pady=10)
# 创建Label用于显示视频流
label_img = tk.Label(root)
label_img.pack()
# 创建多行文本框
text_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=9)  # 设置宽度和高度
text_box.pack(side=tk.BOTTOM, padx=10, pady=10)


# 在窗口关闭时释放资源
def on_closing():
    global cap
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)

# 运行主循环
root.mainloop()