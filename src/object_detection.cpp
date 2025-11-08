#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <deque>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

struct Options {
    std::string device;
    std::string sourceElement;
    int width = 640;
    int height = 480;
    int framerate = 30;
    bool noDisplay = false;
    std::string modelPath;
    std::string configPath;
    int classId = 41;  // default to cup
    float confidence = 0.5f;
    float nms = 0.45f;
    int inputWidth = 640;
    int inputHeight = 640;
    double bufferSeconds = 3.0;
    double recordSeconds = 3.0;
    std::filesystem::path outputDir = "output_clips";
    int bitrateKbps = 4000;
};

struct Detection {
    cv::Rect box;
    int classId;
    float confidence;
};

static void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "Options:\n"
              << "  --device VALUE             Camera device (e.g. /dev/video0)\n"
              << "  --source-element GST       Full GStreamer source element segment\n"
              << "  --width N                  Frame width (default 640)\n"
              << "  --height N                 Frame height (default 480)\n"
              << "  --framerate N              Frame rate (default 30)\n"
              << "  --no-display               Disable OpenCV preview window\n"
              << "  --model PATH               Path to DNN model (.onnx, .pb, etc.) [required]\n"
              << "  --config PATH              Optional DNN config (.pbtxt, .cfg, etc.)\n"
              << "  --class-id N               Target class id to record (default 41)\n"
              << "  --confidence FLOAT         Confidence threshold (default 0.5)\n"
              << "  --nms-threshold FLOAT      NMS IoU threshold (default 0.45)\n"
              << "  --input-size W H           DNN input size (default 640 640)\n"
              << "  --buffer-seconds FLOAT     Seconds to buffer before trigger (default 3)\n"
              << "  --record-seconds FLOAT     Seconds to record after trigger (default 3)\n"
              << "  --output-dir PATH          Directory for recorded clips (default ./output_clips)\n"
              << "  --bitrate N                Recording bitrate kbps (default 4000)\n"
              << "  -h, --help                 Show this help message\n";
}

static bool parse_args(int argc, char** argv, Options& opts) {
    if (argc == 1) {
        print_usage(argv[0]);
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };

        try {
            if (arg == "--device") {
                opts.device = require_value(arg);
            } else if (arg == "--source-element") {
                opts.sourceElement = require_value(arg);
            } else if (arg == "--width") {
                opts.width = std::stoi(require_value(arg));
            } else if (arg == "--height") {
                opts.height = std::stoi(require_value(arg));
            } else if (arg == "--framerate") {
                opts.framerate = std::stoi(require_value(arg));
            } else if (arg == "--no-display") {
                opts.noDisplay = true;
            } else if (arg == "--model") {
                opts.modelPath = require_value(arg);
            } else if (arg == "--config") {
                opts.configPath = require_value(arg);
            } else if (arg == "--class-id") {
                opts.classId = std::stoi(require_value(arg));
            } else if (arg == "--confidence") {
                opts.confidence = std::stof(require_value(arg));
            } else if (arg == "--nms-threshold") {
                opts.nms = std::stof(require_value(arg));
            } else if (arg == "--input-size") {
                opts.inputWidth = std::stoi(require_value(arg));
                if (i + 1 >= argc) {
                    throw std::runtime_error("Missing second value for --input-size");
                }
                opts.inputHeight = std::stoi(argv[++i]);
            } else if (arg == "--buffer-seconds") {
                opts.bufferSeconds = std::stod(require_value(arg));
            } else if (arg == "--record-seconds") {
                opts.recordSeconds = std::stod(require_value(arg));
            } else if (arg == "--output-dir") {
                opts.outputDir = require_value(arg);
            } else if (arg == "--bitrate") {
                opts.bitrateKbps = std::stoi(require_value(arg));
            } else if (arg == "-h" || arg == "--help") {
                print_usage(argv[0]);
                return false;
            } else {
                std::cerr << "Unknown argument: " << arg << "\n";
                print_usage(argv[0]);
                return false;
            }
        } catch (const std::exception& ex) {
            std::cerr << "Error parsing " << arg << ": " << ex.what() << "\n";
            return false;
        }
    }

    if (opts.modelPath.empty()) {
        std::cerr << "--model is required\n";
        print_usage(argv[0]);
        return false;
    }
    return true;
}

static std::string build_source_segment(const Options& opts) {
    if (!opts.sourceElement.empty()) {
        return opts.sourceElement;
    }

#ifdef _WIN32
    std::string device = opts.device.empty() ? "0" : opts.device;
    return "ksvideosrc device-index=" + device;
#elif defined(__APPLE__)
    std::string device = opts.device.empty() ? "0" : opts.device;
    return "avfvideosrc device-index=" + device;
#else
    std::string device = opts.device.empty() ? "/dev/video0" : opts.device;
    return "v4l2src device=" + device;
#endif
}

class Detector {
public:
    Detector(const Options& opts)
        : confidence_(opts.confidence),
          nms_(opts.nms),
          filterClassId_(opts.classId),
          inputSize_(opts.inputWidth, opts.inputHeight) {
        const std::string modelExt = std::filesystem::path(opts.modelPath).extension().string();
        if (modelExt == ".onnx") {
            net_ = cv::dnn::readNetFromONNX(opts.modelPath);
            isYolo_ = true;
        } else {
            if (opts.configPath.empty()) {
                detector_ = cv::dnn::DetectionModel(opts.modelPath);
            } else {
                detector_ = cv::dnn::DetectionModel(opts.modelPath, opts.configPath);
            }
            detector_.setInputSize(inputSize_);
            detector_.setInputScale(1.0 / 127.5);
            detector_.setInputMean(cv::Scalar(127.5, 127.5, 127.5));
            detector_.setInputSwapRB(true);
            isYolo_ = false;
        }
    }

    std::vector<Detection> detect(const cv::Mat& frame) {
        if (isYolo_) {
            return detectYolo(frame);
        }
        return detectGeneric(frame);
    }

private:
    std::vector<Detection> detectYolo(const cv::Mat& frame) {
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, inputSize_, cv::Scalar(), true, false);
        net_.setInput(blob);
        cv::Mat outputs = net_.forward();

        const int dims = outputs.dims;
        if (dims != 3 && dims != 4) {
            return {};
        }

        const int64_t numProposals = (dims == 3) ? outputs.size[2] : outputs.size[1];
        const int64_t numAttributes = (dims == 3) ? outputs.size[1] : outputs.size[2];
        const int numClasses = static_cast<int>(numAttributes - 5);

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> classIds;

        for (int64_t i = 0; i < numProposals; ++i) {
            float x = outputs.at<float>(0, 0, i);
            float y = outputs.at<float>(0, 1, i);
            float w = outputs.at<float>(0, 2, i);
            float h = outputs.at<float>(0, 3, i);
            float objectness = outputs.at<float>(0, 4, i);
            if (objectness < confidence_) {
                continue;
            }

            int bestClass = -1;
            float bestClassScore = 0.0f;
            for (int c = 0; c < numClasses; ++c) {
                float score = outputs.at<float>(0, static_cast<int>(5 + c), i);
                if (score > bestClassScore) {
                    bestClassScore = score;
                    bestClass = c;
                }
            }

            float confidence = objectness * bestClassScore;
            if (confidence < confidence_) {
                continue;
            }

            int classId = bestClass;
            float cx = x * frame.cols;
            float cy = y * frame.rows;
            float boxW = w * frame.cols;
            float boxH = h * frame.rows;
            int left = static_cast<int>(std::round(cx - boxW / 2));
            int top = static_cast<int>(std::round(cy - boxH / 2));
            int width = static_cast<int>(std::round(boxW));
            int height = static_cast<int>(std::round(boxH));
            if (width <= 0 || height <= 0) {
                continue;
            }

            boxes.emplace_back(left, top, width, height);
            confidences.push_back(confidence);
            classIds.push_back(classId);
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confidence_, nms_, indices);

        std::vector<Detection> detections;
        for (int idx : indices) {
            if (filterClassId_ >= 0 && classIds[idx] != filterClassId_) {
                continue;
            }
            detections.push_back({boxes[idx], classIds[idx], confidences[idx]});
        }
        return detections;
    }

    std::vector<Detection> detectGeneric(const cv::Mat& frame) {
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        detector_.detect(frame, classIds, confidences, boxes, confidence_, nms_);

        std::vector<Detection> detections;
        for (size_t i = 0; i < boxes.size(); ++i) {
            if (boxes[i].width <= 0 || boxes[i].height <= 0) {
                continue;
            }
            if (filterClassId_ >= 0 && classIds[i] != filterClassId_) {
                continue;
            }
            detections.push_back({boxes[i], classIds[i], confidences[i]});
        }
        return detections;
    }

    bool isYolo_ = false;
    float confidence_;
    float nms_;
    int filterClassId_;
    cv::Size inputSize_;
    cv::dnn::Net net_;
    cv::dnn::DetectionModel detector_;
};

class RecordingSink {
public:
    RecordingSink(int width, int height, int fps, int bitrateKbps, const std::filesystem::path& outputPath)
        : frameDuration_(fps > 0 ? GST_SECOND / fps : GST_SECOND),
          outputPath_(outputPath) {
        std::string pipelineDesc =
            "appsrc name=recsrc is-live=false block=true format=time "
            "caps=video/x-raw,format=BGR,width=" +
            std::to_string(width) + ",height=" + std::to_string(height) + ",framerate=" + std::to_string(fps) + "/1 ! "
            "videoconvert ! video/x-raw,format=I420 ! "
            "x264enc tune=zerolatency speed-preset=ultrafast bitrate=" + std::to_string(bitrateKbps) + " key-int-max=" +
            std::to_string(fps) + " ! "
            "mp4mux ! filesink name=recfilesink";

        GError* error = nullptr;
        pipeline_ = gst_parse_launch(pipelineDesc.c_str(), &error);
        if (!pipeline_) {
            std::string message = "Failed to create recording pipeline";
            if (error) {
                message += ": " + std::string(error->message);
                g_error_free(error);
            }
            throw std::runtime_error(message);
        }

        GstElement* element = gst_bin_get_by_name(GST_BIN(pipeline_), "recsrc");
        appsrc_ = GST_APP_SRC(element);
        if (!appsrc_) {
            gst_object_unref(pipeline_);
            throw std::runtime_error("Failed to get appsrc from recording pipeline");
        }

        GstElement* filesink = gst_bin_get_by_name(GST_BIN(pipeline_), "recfilesink");
        g_object_set(filesink, "location", outputPath.string().c_str(), nullptr);
        gst_object_unref(filesink);

        g_object_set(appsrc_, "do-timestamp", FALSE, "format", GST_FORMAT_TIME, nullptr);
        bus_ = gst_element_get_bus(pipeline_);
        gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    }

    ~RecordingSink() {
        if (pipeline_) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }
    }

    void push_frame(const cv::Mat& frame, GstClockTime pts) {
        cv::Mat continuousFrame = frame;
        if (!frame.isContinuous()) {
            continuousFrame = frame.clone();
        }

        const size_t size = static_cast<size_t>(continuousFrame.total() * continuousFrame.elemSize());
        GstBuffer* buffer = gst_buffer_new_allocate(nullptr, size, nullptr);
        GstMapInfo map;
        gst_buffer_map(buffer, &map, GST_MAP_WRITE);
        std::memcpy(map.data, continuousFrame.data, size);
        gst_buffer_unmap(buffer, &map);

        GST_BUFFER_PTS(buffer) = pts;
        GST_BUFFER_DTS(buffer) = pts;
        GST_BUFFER_DURATION(buffer) = frameDuration_;

        GstFlowReturn ret = gst_app_src_push_buffer(appsrc_, buffer);
        if (ret != GST_FLOW_OK) {
            std::cerr << "[record] Warning: failed to push buffer (flow="
                      << gst_flow_get_name(ret) << ")\n";
        }
    }

    void finish(guint64 timeoutNs = 5 * GST_SECOND) {
        gst_app_src_end_of_stream(appsrc_);
        if (bus_) {
            gst_bus_timed_pop_filtered(bus_, timeoutNs, static_cast<GstMessageType>(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
            gst_object_unref(bus_);
            bus_ = nullptr;
        }
        gst_element_set_state(pipeline_, GST_STATE_NULL);
    }

private:
    GstElement* pipeline_ = nullptr;
    GstAppSrc* appsrc_ = nullptr;
    GstBus* bus_ = nullptr;
    GstClockTime frameDuration_;
    std::filesystem::path outputPath_;
};

static cv::Mat sample_to_mat(GstSample* sample, GstClockTime& pts) {
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
        return {};
    }
    pts = GST_BUFFER_PTS(buffer);

    GstCaps* caps = gst_sample_get_caps(sample);
    GstStructure* structure = gst_caps_get_structure(caps, 0);
    int width = 0, height = 0;
    gst_structure_get_int(structure, "width", &width);
    gst_structure_get_int(structure, "height", &height);

    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        return {};
    }
    cv::Mat frame(height, width, CV_8UC3, map.data);
    cv::Mat copy = frame.clone();
    gst_buffer_unmap(buffer, &map);
    return copy;
}

int main(int argc, char** argv) {
    Options opts;
    if (!parse_args(argc, argv, opts)) {
        return 1;
    }

    gst_init(&argc, &argv);

    try {
        Detector detector(opts);
        std::filesystem::create_directories(opts.outputDir);

        std::string sourceSegment = build_source_segment(opts);
        std::string pipelineDesc = sourceSegment + " ! "
            "videoconvert ! video/x-raw,format=BGR,width=" + std::to_string(opts.width) +
            ",height=" + std::to_string(opts.height) + ",framerate=" + std::to_string(opts.framerate) + "/1 ! "
            "appsink name=appsink emit-signals=false sync=false max-buffers=1 drop=true";

        GError* error = nullptr;
        GstElement* pipeline = gst_parse_launch(pipelineDesc.c_str(), &error);
        if (!pipeline) {
            std::string msg = "Failed to create pipeline";
            if (error) {
                msg += ": " + std::string(error->message);
                g_error_free(error);
            }
            throw std::runtime_error(msg);
        }

        GstElement* appsinkElement = gst_bin_get_by_name(GST_BIN(pipeline), "appsink");
        GstAppSink* appsink = GST_APP_SINK(appsinkElement);
        if (!appsink) {
            gst_object_unref(pipeline);
            throw std::runtime_error("Failed to get appsink");
        }

        gst_element_set_state(pipeline, GST_STATE_PLAYING);
        GstBus* bus = gst_element_get_bus(pipeline);

        GstClockTime frameDuration = opts.framerate > 0 ? GST_SECOND / opts.framerate : GST_SECOND;
        GstClockTime bufferWindow = static_cast<GstClockTime>(opts.bufferSeconds * GST_SECOND);
        std::deque<std::pair<GstClockTime, cv::Mat>> prebuffer;
        const size_t bufferCapacity = static_cast<size_t>(opts.bufferSeconds * opts.framerate) + 2;
        GstClockTime fallbackPts = 0;

        std::unique_ptr<RecordingSink> recordingSink;
        std::optional<GstClockTime> recordingEndPts;

        auto startRecording = [&](GstClockTime pts) {
        auto now = std::chrono::system_clock::now();
        std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", std::localtime(&nowTime));
        std::filesystem::path clipPath = opts.outputDir / ("clip_" + std::string(buf) + ".mp4");
            std::cout << "[record] Starting capture to " << clipPath << "\n";
            recordingSink = std::make_unique<RecordingSink>(opts.width, opts.height, opts.framerate, opts.bitrateKbps, clipPath);

            GstClockTime cutoff = pts > bufferWindow ? pts - bufferWindow : 0;
            for (const auto& entry : prebuffer) {
                if (entry.first >= cutoff) {
                    recordingSink->push_frame(entry.second, entry.first);
                }
            }
            recordingEndPts = pts + static_cast<GstClockTime>(opts.recordSeconds * GST_SECOND);
        };

        std::string windowName = "GStreamer Object Detection";

        bool running = true;
        while (running) {
            std::cout << "[loop] waiting for sample..." << std::endl;
            GstSample* sample = gst_app_sink_try_pull_sample(appsink, GST_SECOND / 5);
            if (!sample) {
                std::cout << "[loop] no sample pulled within timeout" << std::endl;
                GstMessage* msg = gst_bus_timed_pop_filtered(bus, 0, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
                if (msg) {
                    switch (GST_MESSAGE_TYPE(msg)) {
                        case GST_MESSAGE_ERROR: {
                            GError* err;
                            gchar* debug;
                            gst_message_parse_error(msg, &err, &debug);
                            std::cerr << "GStreamer error: " << err->message << "\n";
                            g_error_free(err);
                            g_free(debug);
                            running = false;
                            break;
                        }
                        case GST_MESSAGE_EOS:
                            running = false;
                            break;
                        default:
                            break;
                    }
                    gst_message_unref(msg);
                }
                continue;
            }

            GstClockTime pts = GST_BUFFER_PTS(gst_sample_get_buffer(sample));
            if (pts == GST_CLOCK_TIME_NONE) {
                pts = fallbackPts;
            } else {
                fallbackPts = pts;
            }

            cv::Mat frame = sample_to_mat(sample, pts);
            gst_sample_unref(sample);
            if (frame.empty()) {
                std::cout << "[loop] empty frame" << std::endl;
                continue;
            }

            fallbackPts = pts + frameDuration;

            if (prebuffer.size() >= bufferCapacity) {
                prebuffer.pop_front();
            }
            prebuffer.emplace_back(pts, frame);

            std::vector<Detection> detections = detector.detect(frame);
            if (detections.empty()) {
                std::cout << "[loop] no detections on this frame" << std::endl;
            }
            bool triggered = false;
            for (const auto& detection : detections) {
                std::cout << "Detection: class_id=" << detection.classId
                          << ", confidence=" << detection.confidence
                          << ", box=(" << detection.box.x << ", " << detection.box.y
                          << ", " << detection.box.width << ", " << detection.box.height << ")\n";

                cv::rectangle(frame, detection.box, cv::Scalar(0, 255, 0), 2);
                std::string label = "class " + std::to_string(detection.classId);
                if (opts.classId >= 0 && detection.classId == opts.classId) {
                    label = "target " + std::to_string(detection.confidence);
                } else {
                    label += " " + cv::format("%.2f", detection.confidence);
                }
                cv::putText(frame, label, cv::Point(detection.box.x, std::max(0, detection.box.y - 10)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

                if (opts.classId < 0 || detection.classId == opts.classId) {
                    triggered = true;
                }
            }

            if (triggered && !recordingSink) {
                startRecording(pts);
            }

            if (recordingSink) {
                recordingSink->push_frame(frame, pts);
                if (recordingEndPts && pts >= *recordingEndPts) {
                    recordingSink->finish();
                    std::cout << "[record] Completed clip at pts=" << pts << "\n";
                    recordingSink.reset();
                    recordingEndPts.reset();
                }
            }

            if (!opts.noDisplay) {
                cv::imshow(windowName, frame);
                if (cv::waitKey(1) == 'q') {
                    break;
                }
            }
        }

        if (recordingSink) {
            recordingSink->finish();
        }

        if (!opts.noDisplay) {
            cv::destroyAllWindows();
        }

        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(bus);
        gst_object_unref(pipeline);

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}

