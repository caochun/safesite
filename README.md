# SafeSite 实时目标检测演示

本项目演示如何使用 GStreamer 采集摄像头画面，结合 OpenCV DNN 推理 YOLOv8n 模型，实现茶杯（COCO `class_id=41`）检测，并在触发时回溯保存前三秒的视频片段。

> 💡 **Rockchip 平台提示**  
> RK3588 等 SoC 内置基于 MPP 的 GStreamer 插件（如 `mppvideodec`、`mpph264enc`、`rkximagesink`、`rkisp`），可提供硬件编解码、零拷贝渲染与 ISP 能力。建议直接使用 Rockchip 官方 SDK/镜像中的 `gstreamer-rockchip` 套件，并通过 `gst-inspect-1.0` 检查插件和驱动是否正确加载。

## 快速开始

1. 准备环境
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt  # 如果没有 requirements.txt，请安装 opencv-python numpy pygobject
   ```

   - **macOS**（Homebrew）：
     ```bash
     brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-libav pygobject3 gtk4
     ```

   - **Ubuntu / Debian 系列**：
     ```bash
     sudo apt update
     sudo apt install \
       gstreamer1.0-tools \
       gstreamer1.0-plugins-base \
       gstreamer1.0-plugins-good \
       gstreamer1.0-plugins-bad \
       gstreamer1.0-plugins-ugly \
       gstreamer1.0-libav \
       python3-gi \
       gir1.2-gstreamer-1.0 \
       gir1.2-gtk-3.0 \
       libglib2.0-dev
     ```
     根据需求可追加硬件相关插件（如 VAAPI、NVIDIA）。
   ```

2. 下载 YOLOv8n ONNX 模型  
   使用 Hugging Face 提供的权重文件：
   ```
   https://huggingface.co/SpotLab/YOLOv8Detection/resolve/3005c6751fb19cdeb6b10c066185908faf66a097/yolov8n.onnx
   ```
   将文件放置到 `models/yolov8n.onnx`。

3. 运行脚本
   ```bash
   python scripts/object_detection.py \
     --model models/yolov8n.onnx \
     --class-id 41 \
     --confidence 0.25 \
     --input-size 640 640 \
     --buffer-seconds 3 \
     --record-seconds 3
   ```
   - 触发检测时会在 `output_clips/` 生成包含触发前后视频片段的 MP4。
   - 如果不带参数运行脚本，会自动打印使用帮助。

4. 常见参数
   - `--device` 指定摄像头设备（如 `/dev/video0` 或 `0`）。
   - `--source-element` 自定义 GStreamer 源元素。
   - `--buffer-seconds`、`--record-seconds` 控制回溯和触发后录像时长。
   - `--bitrate` 设置输出视频编码码率（kbps）。

## 注意事项
- 默认为 `cup`（COCO `class_id=41`），需要根据实际模型类别调整。
- `models/` 与 `output_clips/` 已在 `.gitignore` 中忽略，无法直接上传到仓库。
- 若 `gi` 或 GStreamer 库缺失，请先安装对应的系统依赖。

## 参考链接
- YOLOv8n ONNX 下载：[SpotLab/YOLOv8Detection](https://huggingface.co/SpotLab/YOLOv8Detection/resolve/3005c6751fb19cdeb6b10c066185908faf66a097/yolov8n.onnx)

