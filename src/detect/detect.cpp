//
// Created by bruce on 2020/10/16.
//

#include "detect.h"

float iou(float lbox[4], float rbox[4])
{
    float interBox[] = {
            (std::max)(lbox[0] - lbox[2] / 2.f, rbox[0] - rbox[2] / 2.f), //left
            (std::min)(lbox[0] + lbox[2] / 2.f, rbox[0] + rbox[2] / 2.f), //right
            (std::max)(lbox[1] - lbox[3] / 2.f, rbox[1] - rbox[3] / 2.f), //top
            (std::min)(lbox[1] + lbox[3] / 2.f, rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] or interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

void nms(vector<DETECT::DET_BOX> &bboxes, float *output, float conf_thresh, float nms_thresh = 0.5)
{
    // 1st step, parse the raw output to a map
    int det_size = sizeof(Yolo::Detection) / sizeof(float);
    std::map<float, std::vector<Yolo::Detection>> m; // class -- obj list
    for (int i = 0; i < int(output[0]) and i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) // output[0] may be the number of ROIs
    {
        if (output[1 + det_size * i + 4] <= conf_thresh)
            // 0 is the number, so start from 1
            // det_size * i is the start index of bbox
            // 4 is the offset of conf
            continue;
        Yolo::Detection det{};
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));

        // if no obj of this class was detected before, create a obj list
        if (m.count(det.class_id) == 0)
            m.emplace(det.class_id, std::vector<Yolo::Detection>());
        // the obj list of this class append this obj
        m[det.class_id].emplace_back(det);
    }

    // 2nd step, nms
    for (auto &it : m)
    {
        auto &dets = it.second;
        sort(dets.begin(), dets.end(), [](Yolo::Detection &a, Yolo::Detection &b) { return a.conf > b.conf; });
        for (size_t i = 0; i < dets.size(); i++)
        {
            auto &item = dets[i];

            auto cx = item.bbox[0];
            auto cy = item.bbox[1];
            auto w = item.bbox[2];
            auto h = item.bbox[3];
            bboxes.emplace_back(DETECT::DET_BOX{int(item.class_id),
                                                int(cx - w / 2), int(cy - h / 2),
                                                int(cx + w / 2), int(cy + h / 2),
                                                item.conf});

            for (size_t n = i + 1; n < dets.size(); n++)
            {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh)
                {
                    dets.erase(dets.begin() + n);
                    n--;
                }
            }
        }
    }

}

DETECT::DETECT()
{
    _inputH = Yolo::INPUT_H;
    _inputW = Yolo::INPUT_W;
    _inputSize = _inputC * _inputW * _inputH;
    _outputSize = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;
}

DETECT::~DETECT() = default;

int DETECT::initFromEngine(string &modelPath, int device)
{
    int ret;
    ifstream file(modelPath, ios::binary);
    char *trtModelStream;
    size_t fsize;
    if (file.good())
    {
        file.seekg(0, ifstream::end);
        fsize = file.tellg();
        file.seekg(0, ifstream::beg);
        trtModelStream = new char[fsize];
        file.read(trtModelStream, fsize);
        file.close();
    }
    else
    {
        cerr << "[ERROR] Failed to load model " << modelPath << endl;
        return -1;
    }

    ret = initFromEngine(trtModelStream, fsize, device);
    if (ret != 0) return ret;

    delete[] trtModelStream;
    return 0;
}

int DETECT::initFromEngine(char *modelPtr, size_t slFileSize, int device)
{
    int ret;
    cudaSetDevice(device);
    _gLogger = new Logger();
    _runtime = nvinfer1::createInferRuntime(*_gLogger);
    CHECK_EXPR_RET(_runtime == nullptr, "createInferRuntime failed", -1)
    _engine = _runtime->deserializeCudaEngine(modelPtr, slFileSize);
    CHECK_EXPR_RET(_engine == nullptr, "deserializeCudaEngine failed", -1)
    _context = _engine->createExecutionContext();
    CHECK_EXPR_RET(_context == nullptr, "createExecutionContext failed", -1)
    CHECK_EXPR_RET(_engine->getNbBindings() != 2, "getNbBindings is not 2", -1)

    _inputIndex = _engine->getBindingIndex("data");
    CHECK_EXPR_RET(_inputIndex != 0, "_inputIndex is not 0", -1)
    _outputIndex = _engine->getBindingIndex("prob");
    CHECK_EXPR_RET(_inputIndex != 0, "_outputIndex is not 1", -1)

    ret = cudaMalloc(&_buffers[_inputIndex], _inputSize * sizeof(float));
    CHECK_EXPR_RET(ret != 0, "cudaMalloc failed", -1)
    ret = cudaMalloc(&_buffers[_outputIndex], _outputSize * sizeof(float));
    CHECK_EXPR_RET(ret != 0, "cudaMalloc failed", -1)

    _cudaStream = new cudaStream_t;
    ret = cudaStreamCreate(_cudaStream);
    CHECK_EXPR_RET(ret != 0, "cudaStreamCreate failed", -1)
}

int DETECT::forward(string &picPath)
{
    float data[_inputSize];
    auto img = cv::imread(picPath);
    cv::resize(img, img, cv::Size(_inputW, _inputH));
    int i = 0;
    for (int row = 0; row < _inputH; row++)
    {
        uchar *uc_pixel = img.data + row * img.step;
        for (int col = 0; col < _inputW; col++)
        {
            data[i + _inputH * _inputW * 0] = (float) uc_pixel[2] / 255.f;
            data[i + _inputH * _inputW * 1] = (float) uc_pixel[1] / 255.f;
            data[i + _inputH * _inputW * 2] = (float) uc_pixel[0] / 255.f;
            uc_pixel += 3;
            i++;
        }
    }

    auto ret = forward(data, _inputSize);
    CHECK_EXPR_RET(ret != 0, "forward failed", ret)

    return 0;
}

int DETECT::forward(cv::Mat &img)
{
    float data[_inputSize];
    int i = 0;
    for (int row = 0; row < _inputH; row++)
    {
        uchar *uc_pixel = img.data + row * img.step;
        for (int col = 0; col < _inputW; col++)
        {
            data[i + _inputH * _inputW * 0] = (float) uc_pixel[2] / 255.f;
            data[i + _inputH * _inputW * 1] = (float) uc_pixel[1] / 255.f;
            data[i + _inputH * _inputW * 2] = (float) uc_pixel[0] / 255.f;
            uc_pixel += 3;
            i++;
        }
    }
    auto ret = forward(data, _inputSize);
    CHECK_EXPR_RET(ret != 0, "forward failed", ret)

    return 0;
}

int DETECT::forward(float *data, size_t dataLen)
{
    CHECK_EXPR_RET(dataLen != _inputSize, "Input data size is invalid", -1)
    int iRet;
    iRet = cudaMemcpyAsync(_buffers[0], data, _inputSize * sizeof(float),
                           cudaMemcpyHostToDevice, *_cudaStream);
    CHECK_EXPR_RET(iRet != 0, "cudaMemcpyAsync failed", -1)
    auto bRet = _context->enqueue(1, _buffers, *_cudaStream, nullptr);
    CHECK_EXPR_RET(bRet == false, "enqueue failed", -1)
    float prob[_outputSize];
    iRet = cudaMemcpyAsync(prob, _buffers[1], _outputSize * sizeof(float),
                           cudaMemcpyDeviceToHost, *_cudaStream);
    CHECK_EXPR_RET(iRet != 0, "cudaMemcpyAsync failed", -1)
    iRet = cudaStreamSynchronize(*_cudaStream);
    CHECK_EXPR_RET(iRet != 0, "cudaStreamSynchronize failed", -1)

    bboxes.clear();
    bboxes.shrink_to_fit();
    nms(bboxes, prob, confThresh, nmsThresh);

    return 0;
}

int DETECT::Deinit()
{
    int ret;
    ret = cudaStreamDestroy(*_cudaStream);
    CHECK_EXPR_RET(ret != 0, "cudaStreamDestroy failed", -1)
    ret = cudaFree(_buffers[_inputIndex]);
    CHECK_EXPR_RET(ret != 0, "cudaFree input failed", -1)
    ret = cudaFree(_buffers[_outputIndex]);
    CHECK_EXPR_RET(ret != 0, "cudaFree output failed", -1)

    _context->destroy();
    _engine->destroy();
    _runtime->destroy();

    delete _gLogger;
    delete _cudaStream;

    return 0;
}
