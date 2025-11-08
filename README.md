# SafeSite 实时目标检测演示

本项目演示如何使用 GStreamer 采集摄像头画面，结合 OpenCV DNN 推理 YOLOv8n 模型，实现茶杯（COCO `class_id=41`）检测，并在触发时回溯保存前三秒的视频片段。

> 💡 **Rockchip 平台提示**  
> RK3588 等 SoC 内置基于 MPP 的 GStreamer 插件（如 `mppvideodec`、`mpph264enc`、`rkximagesink`、`rkisp`），可提供硬件编解码、零拷贝渲染与 ISP 能力。建议直接使用 Rockchip 官方 SDK/镜像中的 `gstreamer-rockchip` 套件，并通过 `gst-inspect-1.0` 检查插件和驱动是否正确加载。

## 快速开始

1. 安装依赖
   - **macOS（Homebrew）**
     ```bash
     brew install cmake opencv gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-libav
     ```

   - **Ubuntu / Debian 系列**
     ```bash
     sudo apt update
     sudo apt install \
       build-essential cmake pkg-config \
       libopencv-dev \
       gstreamer1.0-tools \
       gstreamer1.0-plugins-base \
       gstreamer1.0-plugins-good \
       gstreamer1.0-plugins-bad \
       gstreamer1.0-plugins-ugly \
       gstreamer1.0-libav \
       libgstreamer1.0-dev \
       libgstreamer-plugins-base1.0-dev
     ```
     根据硬件需求追加 VAAPI、NVIDIA 或 Rockchip 相关插件。

2. 下载 YOLOv8n ONNX 模型  
   使用 Hugging Face 提供的权重文件（需先 `mkdir -p models`）：
   ```
   https://huggingface.co/SpotLab/YOLOv8Detection/resolve/3005c6751fb19cdeb6b10c066185908faf66a097/yolov8n.onnx
   ```

3. 构建原生应用
   ```bash
   cmake -S . -B build
   cmake --build build
   ```

4. 运行示例
   ```bash
   ./build/object_detection \
     --model models/yolov8n.onnx \
     --class-id 41 \
     --confidence 0.25 \
     --input-size 640 640 \
     --buffer-seconds 3 \
     --record-seconds 3
   ```
   - 触发检测时会在 `output_clips/` 生成包含触发前后画面的 MP4 文件。
   - 不带参数执行会打印使用说明。

5. 常见参数
   - `--device` 指定摄像头设备（如 `/dev/video0`、`0`）。
   - `--source-element` 自定义 GStreamer 源元素（如 RTSP、RTMP）。
   - `--buffer-seconds`、`--record-seconds` 控制回溯与触发后保存时长。
   - `--bitrate` 设置录制编码码率（kbps）。

## 注意事项
- 默认为 `cup`（COCO `class_id=41`），需要根据实际模型类别调整。
- 使用 `x264enc` 进行 H.264 编码，如需硬件或其他编码器，可修改源码中录制管线。
- `models/` 与 `output_clips/` 已在 `.gitignore` 中忽略，需自行创建相应目录。
- 构建前请确保系统已安装 GStreamer/OpenCV 开发包以及 `pkg-config`。

## 参考链接
- YOLOv8n ONNX 下载：[SpotLab/YOLOv8Detection](https://huggingface.co/SpotLab/YOLOv8Detection/resolve/3005c6751fb19cdeb6b10c066185908faf66a097/yolov8n.onnx)

