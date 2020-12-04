//
// Created by ljh on 2020/10/23.
//
#ifndef FINAL_YOLOV5_SVIP_INTERFACE_H
#define FINAL_YOLOV5_SVIP_INTERFACE_H

#pragma once

#include <fstream>
#include "track.h"
#include <detect/detect.h>
#include "ActRecognize.h"
#include <detect/crop_util.h>
#include "md5/md5.h"
#include "svipAISDK_V3.h"
#include "opencv2/opencv.hpp"
#include "PoseEst.h"

#include <torch/script.h> // One-stop header.
#include <vector>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
//#include "skeleton_seq_preprocess.h"
#include <fstream>

using namespace std;
using namespace cv;
using namespace std;
using namespace torch;

class BRIDGE
{
public:
    long long mytime = 0;
    ApplicationType applicationType{};
    CameraType cameraType{};
    svip_ai_result_cb mycallback{};
    svip_ai_error_cb svip_ai_error_cb_ = nullptr;
    void * svip_ai_error_cb_user_ = nullptr;
    void *myuser = nullptr;
    void *_params = nullptr;

    DETECT          *detector      = nullptr;
    TRACK           *tracker       = nullptr;
    PoseEst         *poseEst       = nullptr;

    int oriWidth, oriHeight;

//    int setFrame(AstAIFrame *frameInfo);
//
//    int checkMD5(FrameInfo *frameInfo, const char *md5_);
//
//    int checkFMT(FrameInfo *frameInfo) const;
    int setFrame(AstAIFrame *frameInfo);

    int checkMD5(FrameInfo *frameInfo, const char *md5_);

    int checkFMT(FrameInfo *frameInfo) const;

    cv::Mat preprocessImg(cv::Mat &src, FrameInfo *frameInfo);
    int runFrame(const char *md5_, FrameInfo *frame_info);

    int parseDetRes(vector<DETECT::DET_BOX> &srcBoxes, vector<DETECT::DET_BOX> &dstBoxes) const;

    int parse(FrameInfo *frameInfo);
    int Deinit() const;


    // 这个vector保存着一个视频帧里面的vector，如(50, 17, 2)
    vector<vector<vector<int>>> _kps_seq;
    // 这个vector保存着每一张图片的坐标或者是别的hourglass的输出，如(17, 2)
    vector<vector<int>> _kps;
    // 这个vector用于动态load，当图片解码之后，送入这个vector，然后deal_frame还要用到这个向量，用来得到坐标呢
    vector<Mat> _pic_candidate;

    //// block_left和block_right组成了所有的内存块
    vector<vector<pair<int, Mat>>> block_left;
    vector<vector<pair<int, Mat>>> block_right;

    //// 同一个id下的block
    vector<pair<int, Mat>> prev_block;
    vector<pair<int, Mat>> curr_block;

    int _index = 999;

    vector<int> pre_IDs;


private:
    float _rw, _rh;
    int _re_w, _re_h, _re_x, _re_y;
    // 将pose提取出来的关节点坐标，送入该vector
    vector<float> coordinate_xy;
//    std::unique_ptr<tflite::FlatBufferModel> model;
    MD5_CTX md5{};
};



//class ACT_BRIDGE {
//public:
//    ActRecognizer *actRecognizer = nullptr;
//
//
//};
//
//class DYNAMIC {
//    int dynamic_save(vector<int> & pre_IDs, pair<int, Mat> & curr_pair);
//    int dynamic_load(vector<vector<pair<int, Mat>>> block_left);
//
//// 这个vector保存着一个视频帧里面的vector，如(50, 17, 2)
//    vector<vector<vector<int>>> _kps_seq;
//// 这个vector保存着每一张图片的坐标或者是别的hourglass的输出，如(17, 2)
//    vector<vector<int>> _kps;
//// 这个vector用于动态load，当图片解码之后，送入这个vector，然后deal_frame还要用到这个向量，用来得到坐标呢
//    vector<Mat> _pic_candidate;
//
////// block_left和block_right组成了所有的内存块
//    vector<vector<pair<int, Mat>>> block_left;
//    vector<vector<pair<int, Mat>>> block_right;
//
////// 同一个id下的block
//    vector<pair<int, Mat>> prev_block;
//    vector<pair<int, Mat>> curr_block;
//
//    int _index = 999;
//
//    vector<int> pre_IDs;
//};



#endif //FINAL_YOLOV5_SVIP_INTERFACE_H
