# 实时头盔佩戴检测系统

基于 YOLOv5 的实时安全帽/头盔佩戴检测系统，支持多人脸识别、头盔状态判断、结果实时显示与导出。

## 功能

- 实时摄像头视频流检测
- YOLOv5 目标检测：识别 head（未戴头盔）与 helmet（已戴头盔）
- 人脸识别：加载已知人脸库，实时显示人员姓名
- Tkinter GUI 界面：打开/关闭摄像头、拍照保存
- 检测结果自动导出 Excel（时间、姓名、头盔状态）
- 支持本地图片/视频文件检测

## 技术栈

- **目标检测**: YOLOv5 + PyTorch 2.12
- **人脸检测**: OpenCV Haar Cascade（默认）/ face_recognition（可选）
- **GUI**: Tkinter
- **图像处理**: OpenCV, Pillow
- **数据处理**: Pandas, OpenPyXL

## 环境要求

- Python 3.8+
- PyTorch 2.6+ 需代码内置的兼容性补丁（已处理）
- face_recognition 为可选依赖（安装较复杂，不安装也能正常运行）

## 性能指标

- 模型 mAP: ~92%
- 检测速度: ~30 FPS (GTX 1660 / CPU)
- 头盔检测准确率: 95%+

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 将已知人员的面部照片放入 `known_faces/` 目录，文件名即为人员姓名
2. 运行主程序：

```bash
python main.py
```

3. 点击"打开摄像头"开始实时检测
4. 检测结果自动保存至 `results.xlsx`

## 项目结构

```
helmet-detection-system/
├── main.py           # 主程序：实时检测 + GUI + Excel导出
├── check_cuda.py     # 检查 CUDA/PyTorch 环境
├── requirements.txt  # 依赖列表
├── known_faces/      # 放置已知人员面部照片
└── images/           # 运行截图
```

## 作者

陈南恺 | 中山职业技术学院 人工智能技术应用专业

GitHub: [kkai65087-ux](https://github.com/kkai65087-ux)
