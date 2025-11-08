#!/usr/bin/env python3
import argparse
import collections
import datetime
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import gi
import numpy as np

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst  # noqa: E402
from gi.repository import GstApp  # noqa: E402


def build_source_element(args: argparse.Namespace) -> str:
    """
    Return the appropriate video source element segment for the platform.
    """
    if args.source_element:
        return args.source_element

    if sys.platform.startswith("linux"):
        device = args.device or "/dev/video0"
        return f"v4l2src device={device}"
    if sys.platform == "darwin":
        device = args.device or "0"
        return f"avfvideosrc device-index={device}"
    if os.name == "nt":
        device = args.device or "0"
        return f"ksvideosrc device-index={device}"

    raise RuntimeError("Unsupported platform; please specify --source-element explicitly.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use GStreamer + OpenCV to detect cups from a live camera stream."
    )
    parser.add_argument(
        "--device",
        help="Camera device (e.g. /dev/video0 on Linux, index on macOS/Windows).",
    )
    parser.add_argument(
        "--source-element",
        help="Full GStreamer source element segment (e.g. 'v4l2src device=/dev/video2').",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Requested frame width (default: 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Requested frame height (default: 480).",
    )
    parser.add_argument(
        "--framerate",
        type=int,
        default=30,
        help="Requested frame rate numerator (default: 30).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV window display.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a DNN model file (e.g. .onnx, .pb).",
    )
    parser.add_argument(
        "--config",
        help="Path to the DNN model configuration (e.g. .pbtxt). Needed for TensorFlow/Caffe models.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        help="Target class id to detect (e.g. 41 for cup in COCO). Omit to keep all classes.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for detections (default: 0.5).",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.45,
        help="Non-maximum suppression IOU threshold (default: 0.45).",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=(640, 640),
        metavar=("WIDTH", "HEIGHT"),
        help="DNN input resolution (default: 640 640).",
    )
    parser.add_argument(
        "--buffer-seconds",
        type=float,
        default=3.0,
        help="Seconds of video to keep in memory for pre-event buffering (default: 3).",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=3.0,
        help="Seconds of video to save after a trigger (default: 3).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_clips"),
        help="Directory to store captured clips (default: ./output_clips).",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=4000,
        help="Video bitrate in kbps for recorded clips (default: 4000).",
    )
    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()
    return parser.parse_args()


Detection = Tuple[Tuple[int, int, int, int], int, float]


def load_detection_model(args: argparse.Namespace) -> Callable[[np.ndarray], List[Detection]]:
    model_path = Path(args.model)
    width, height = args.input_size

    if model_path.suffix.lower() == ".onnx":
        net = cv2.dnn.readNetFromONNX(str(model_path))

        def detect(frame: np.ndarray) -> List[Detection]:
            blob = cv2.dnn.blobFromImage(
                frame, scalefactor=1.0 / 255.0, size=(width, height), swapRB=True, crop=False
            )
            net.setInput(blob)
            outputs = net.forward()
            outputs = np.squeeze(outputs)
            if outputs.ndim == 2:
                outputs = outputs.T

            frame_h, frame_w = frame.shape[:2]
            detections: List[Detection] = []
            boxes_list: List[List[int]] = []
            confidences: List[float] = []
            class_ids: List[int] = []

            for det in outputs:
                if det.shape[0] < 5:
                    continue
                scores = det[4:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence < args.confidence:
                    continue

                x_center, y_center, box_w, box_h = det[:4]
                x_center *= frame_w
                y_center *= frame_h
                box_w *= frame_w
                box_h *= frame_h

                left = int(round(x_center - box_w / 2))
                top = int(round(y_center - box_h / 2))
                width_px = int(round(box_w))
                height_px = int(round(box_h))

                if width_px <= 0 or height_px <= 0:
                    continue

                boxes_list.append([left, top, width_px, height_px])
                confidences.append(confidence)
                class_ids.append(class_id)

            if boxes_list:
                indices = cv2.dnn.NMSBoxes(boxes_list, confidences, args.confidence, args.nms_threshold)
                if len(indices) > 0:
                    for idx in indices.flatten():
                        box = boxes_list[idx]
                        detections.append((tuple(box), class_ids[idx], confidences[idx]))

            return detections

        return detect

    if args.config:
        net = cv2.dnn_DetectionModel(args.model, args.config)
    else:
        net = cv2.dnn_DetectionModel(args.model)

    net.setInputSize(width, height)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    def detect(frame: np.ndarray) -> List[Detection]:
        classes, scores, boxes = net.detect(frame, confThreshold=args.confidence, nmsThreshold=args.nms_threshold)
        detections: List[Detection] = []
        if classes is not None:
            for class_id, confidence, box in zip(classes.flatten(), scores.flatten(), boxes):
                class_id = int(class_id)
                confidence = float(confidence)
                left, top, w, h = box
                if w <= 0 or h <= 0:
                    continue
                detections.append(((int(left), int(top), int(w), int(h)), class_id, confidence))
        return detections

    return detect


class RecordingSink:
    def __init__(self, width: int, height: int, framerate: int, bitrate_kbps: int, output_path: Path):
        fps = max(framerate, 1)
        self.frame_duration = Gst.SECOND // fps
        self.output_path = output_path
        pipeline_description = (
            "appsrc name=recsrc is-live=false block=true format=time "
            f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
            "videoconvert ! video/x-raw,format=I420 ! "
            f"x264enc tune=zerolatency speed-preset=ultrafast bitrate={bitrate_kbps} key-int-max={fps} ! "
            "mp4mux ! filesink name=recfilesink"
        )
        self.pipeline: Gst.Element = Gst.parse_launch(pipeline_description)
        appsrc_element = self.pipeline.get_by_name("recsrc")
        if not isinstance(appsrc_element, GstApp.AppSrc):
            raise RuntimeError("Failed to acquire appsrc for recording pipeline.")
        self.appsrc: GstApp.AppSrc = appsrc_element
        filesink: Gst.Element = self.pipeline.get_by_name("recfilesink")
        filesink.set_property("location", str(output_path))
        self.appsrc.set_property("do-timestamp", False)
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.pipeline.set_state(Gst.State.PLAYING)
        self.bus = self.pipeline.get_bus()

    def push_frame(self, frame: np.ndarray, pts: int) -> None:
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.pts = pts
        buf.dts = pts
        buf.duration = self.frame_duration
        self.appsrc.emit("push-buffer", buf)

    def finish(self, timeout_ns: int = 5 * Gst.SECOND) -> None:
        self.appsrc.emit("end-of-stream")
        if self.bus:
            self.bus.timed_pop_filtered(timeout_ns, Gst.MessageType.EOS)
        self.pipeline.set_state(Gst.State.NULL)


def gst_buffer_to_ndarray(sample: Gst.Sample) -> Optional[np.ndarray]:
    buffer = sample.get_buffer()
    caps = sample.get_caps()
    structure = caps.get_structure(0)
    width = structure.get_value("width")
    height = structure.get_value("height")

    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        return None

    try:
        frame = np.frombuffer(map_info.data, dtype=np.uint8)
        frame = frame.reshape((height, width, 3))
        return frame.copy()
    finally:
        buffer.unmap(map_info)


def build_pipeline(args: argparse.Namespace) -> tuple[Gst.Pipeline, Gst.Element]:
    source_segment = build_source_element(args)
    pipeline_description = (
        f"{source_segment} ! "
        "videoconvert ! "
        f"video/x-raw,format=BGR,width={args.width},height={args.height},framerate={args.framerate}/1 ! "
        "appsink name=appsink emit-signals=false sync=false max-buffers=1 drop=true"
    )

    pipeline = Gst.parse_launch(pipeline_description)
    appsink = pipeline.get_by_name("appsink")

    if appsink is None:
        raise RuntimeError("Failed to obtain appsink from pipeline.")

    return pipeline, appsink


def main() -> int:
    args = parse_args()
    Gst.init(None)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detect = load_detection_model(args)
    pipeline, appsink = build_pipeline(args)

    pipeline.set_state(Gst.State.PLAYING)
    bus = pipeline.get_bus()

    frame_duration = Gst.SECOND // max(args.framerate, 1)
    buffer_window_ns = int(args.buffer_seconds * Gst.SECOND)
    buffer_capacity = max(int(args.buffer_seconds * args.framerate) + 1, 1)
    prebuffer: collections.deque[Tuple[int, np.ndarray]] = collections.deque(maxlen=buffer_capacity)
    fallback_pts = 0
    recording_sink: Optional[RecordingSink] = None
    recording_end_pts: Optional[int] = None

    def start_recording(current_pts: int) -> RecordingSink:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = args.output_dir / f"clip_{timestamp}.mp4"
        print(f"[record] Starting capture to {output_path}")
        return RecordingSink(args.width, args.height, args.framerate, args.bitrate, output_path)

    window_name = "GStreamer Face Detection"
    try:
        while True:
            sample = appsink.emit("try-pull-sample", Gst.SECOND // 5)
            if sample is None:
                msg = bus.timed_pop_filtered(0, Gst.MessageType.ERROR | Gst.MessageType.EOS)
                if msg:
                    if msg.type == Gst.MessageType.ERROR:
                        err, debug = msg.parse_error()
                        raise RuntimeError(f"GStreamer error: {err.message} (debug: {debug})")
                    break
                continue

            buffer = sample.get_buffer()
            pts = buffer.pts
            if pts == Gst.CLOCK_TIME_NONE:
                pts = fallback_pts
            else:
                fallback_pts = pts

            frame = gst_buffer_to_ndarray(sample)
            if frame is None:
                continue
            raw_frame = frame.copy()
            prebuffer.append((pts, raw_frame))

            fallback_pts = pts + frame_duration

            detections = detect(frame)
            frame_h, frame_w = frame.shape[:2]
            triggered = False
            for (x, y, w, h), class_id, confidence in detections:
                print(f"Detection: class_id={class_id}, confidence={confidence:.3f}, box=({x}, {y}, {w}, {h})")
                if args.class_id is not None and class_id != args.class_id:
                    continue
                x0 = max(0, x)
                y0 = max(0, y)
                x1 = min(frame_w - 1, x + w)
                y1 = min(frame_h - 1, y + h)
                if x1 <= x0 or y1 <= y0:
                    continue
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                if args.class_id is not None and class_id == args.class_id:
                    label_name = "cup"
                else:
                    label_name = f"class {class_id}"
                label = f"{label_name} {confidence:.2f}"
                cv2.putText(frame, label, (x0, max(0, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                if class_id == 41:
                    triggered = True

            recording_started_now = False
            if triggered and recording_sink is None:
                recording_sink = start_recording(pts)
                cutoff_pts = pts - buffer_window_ns
                for stored_pts, stored_frame in prebuffer:
                    if stored_pts >= cutoff_pts:
                        recording_sink.push_frame(stored_frame, stored_pts)
                recording_end_pts = pts + int(args.record_seconds * Gst.SECOND)
                recording_started_now = True

            if recording_sink is not None:
                if not recording_started_now:
                    recording_sink.push_frame(raw_frame, pts)
                if recording_end_pts is not None and pts >= recording_end_pts:
                    recording_sink.finish()
                    print(f"[record] Completed clip; saved until pts={pts}")
                    recording_sink = None
                    recording_end_pts = None

            if not args.no_display:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if recording_sink is not None:
            recording_sink.finish()
            recording_sink = None
        pipeline.set_state(Gst.State.NULL)
        if not args.no_display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

