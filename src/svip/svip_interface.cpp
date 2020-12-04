//
// Created by ljh on 2020/12/2.
//
// todo 注意，在修改的时候，也要把图片给读取出来，然后再加载进去，也就是说中间要添加一步存Mat和读Mat的操作，为什么呢，是因为要检测到根据sort，是否正确提取到一个ID下的数据

#include <svip/svipAISDK_V3.h>

#include "svip_interface.h"

int BRIDGE::checkFMT(FrameInfo *frameInfo) const {
    if (frameInfo->pixel_fmt != PIXEL_FMT_IYUV)
        return SVIP_AI_ERR_PARAMETER;
    return SVIP_AI_OK;
}

int SVIP_AI_Ast_Initialize()
{
    return SVIP_AI_OK;
}

int SVIP_AI_Ast_Uninitialize()
{
    return SVIP_AI_OK;
}

int SVIP_AI_Ast_Start(ApplicationType application_type, CameraType camera_type, void *ai_params, int ai_params_size, svip_ai_result_cb cb, void *user, void **ai_handle) {

    if (APPLICATION_AST != application_type) {
        cerr << "Ast_Start: Application Type Parameter Error" << endl;
        return SVIP_AI_ERR_PARAMETER;
    }

    if (CAMERA_ASSISTANT_ACTION != camera_type) {
        cerr << "Ast_Start: Camera Type Parameter Error" << endl;
        return SVIP_AI_ERR_PARAMETER;
    }

    if (!ai_params || sizeof(AstAIParam) != ai_params_size) {
        cerr << "Ast_Start: AI Parameter Error" << endl;
        return SVIP_AI_ERR_PARAMETER;
    }

    *ai_handle = nullptr;

    auto param = (AstAIParam *) ai_params;

    auto bridge = new BRIDGE;

    if (bridge) {
        string str_model_path = param->path;
        string action_model_best = str_model_path + "/best.engine";

        //count << "Ast_Start: Action Model Best: " << action_model_best.c_str() << endl;
/// 1. detection
        if (!bridge->detector) {
            bridge->detector = new DETECT;
        }
        // 这个阈值，是画框的阈值，而不是类别判断的阈值
        bridge->detector->confThresh = param->confThreshold;
        int nRet = bridge->detector->initFromEngine(action_model_best, 0);
        if (nRet != 0) {
            delete bridge->detector;
            delete bridge;
            cerr << "Ast_Start: Init Error" << endl;
            cerr << "Ast_Start: detection model Error" << endl;
            return SVIP_AI_ERROR;
        }

/// 2. tracker
        if (!bridge->tracker) {
            bridge->tracker = new TRACK;
        }

/// 3. pose estimation
        if (!bridge->poseEst) {
            bridge->poseEst = new PoseEst;
        }

        nRet = bridge->poseEst->pose_init("../model/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite");
        if (nRet != 0) {
            delete bridge->poseEst;
            delete bridge;
            cerr << "Ast_Start: Init Error" << endl;
            cerr << "Ast_Start: pose estimation model Error" << endl;
            return SVIP_AI_ERROR;
        }

        bridge->applicationType = application_type;
        bridge->cameraType = camera_type;
        bridge->mycallback = cb;
//        bridge->actRecognizer->_conf = param->confThreshold;
//        bridge->actRecognizer->_motivation = param->motivation;
//        bridge->actRecognizer->_path = param->path;
        bridge->myuser = user;
        *ai_handle = (void *) bridge;

        return SVIP_AI_OK;
    }
    return SVIP_AI_ERROR;
}


int SVIP_AI_Ast_SetFrame(void *ai_handle, void *ai_frame)
{
    auto ast_ai_frame = ai_frame;
    auto bridge = (BRIDGE *) ai_handle;
    return bridge->setFrame(static_cast<AstAIFrame *>(ast_ai_frame));
}

int SVIP_AI_Ast_InputFrame(void *ai_handle, const char *md5, void *ai_frame)
{
//    auto frame_info = (FrameInfo *) ai_frame;

    auto actionAiFrame = (AstAIFrame*)ai_frame;

    auto frame_info = actionAiFrame->frame_info;

    auto bridge = (BRIDGE *) ai_handle;
//    return bridge->runFrame(md5, frame_info);
    return bridge->runFrame(md5, frame_info);
}

int SVIP_AI_Ast_Stop(void *ai_handle)
{
    auto bridge = (BRIDGE *) ai_handle;
    if (bridge)
    {
        auto ret = bridge->Deinit();
        delete bridge;
        return ret;
    }
    return -1;
}

int BRIDGE::setFrame(AstAIFrame *frameInfo) {

    oriWidth = int(frameInfo->frame_info->width);
    oriHeight = int(frameInfo->frame_info->height);
    return 0;
}


int BRIDGE::checkMD5(FrameInfo *frameInfo, const char *md5_)
{
    char encrypt[50] = {};
    char decrypt[50] = {};
    auto t = frameInfo->timestamp;
    sprintf(encrypt, "bandsoft_%d_%d_%ld", applicationType, cameraType, t);
    MD5Init(&md5);
    MD5Update(&md5, reinterpret_cast<unsigned char *>(encrypt), strlen((char *) encrypt));
    MD5Final(&md5, reinterpret_cast<unsigned char *>(decrypt));
    char out[33] = {0};
    for (size_t i = 0; i < 16; i++)
        sprintf(out + i + i, "%02x", decrypt[i]);
    if (strcasecmp(md5_, out) != 0)
    {
        cerr << "md5 check error. The md5 should be " << out << endl;
        return -7;
    }
    return SVIP_AI_OK;
}

cv::Mat BRIDGE::preprocessImg(cv::Mat &src, FrameInfo *frameInfo)
{
    _rw = Yolo::INPUT_W / float(frameInfo->width);
    _rh = Yolo::INPUT_H / float(frameInfo->height);

    if (_rh > _rw) // original image is a horizontal rectangle
    {
        _re_w = Yolo::INPUT_W;
        _re_h = int(_rw * float(frameInfo->height));
        _re_x = 0;
        _re_y = (Yolo::INPUT_H - _re_h) / 2;
    }
    else // original image is a vertical rectangle
    {
        _re_w = int(_rh * float(frameInfo->width));
        _re_h = Yolo::INPUT_H;
        _re_x = (Yolo::INPUT_W - _re_w) / 2;
        _re_y = 0;
    }
    cv::Mat re(_re_h, _re_w, CV_8UC3);
    cv::resize(src, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(Yolo::INPUT_H, Yolo::INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(_re_x, _re_y, re.cols, re.rows)));
    return out;
}

int BRIDGE::parseDetRes(vector<DETECT::DET_BOX> &srcBoxes, vector<DETECT::DET_BOX> &dstBoxes) const
{
    for (auto &srcBox : srcBoxes)
    {
        auto xmin = int(float(srcBox.xmin - _re_x) / float(_re_w) * float(oriWidth));
        auto ymin = int(float(srcBox.ymin - _re_y) / float(_re_h) * float(oriHeight));
        auto xmax = int(float(srcBox.xmax - _re_x) / float(_re_w) * float(oriWidth));
        auto ymax = int(float(srcBox.ymax - _re_y) / float(_re_h) * float(oriHeight));

        dstBoxes.emplace_back(DETECT::DET_BOX{srcBox.label, xmin, ymin, xmax, ymax, srcBox.conf});
    }
    return 0;
}

int BRIDGE::runFrame(const char *md5_, FrameInfo *frame_info)
{
    //// 来自原始版本的InputFrame
    static int cnt = 0;
    static int mop = 0;
    static int cover = 0;
    static int usher = 0;
    static int iter_mop = 0;
    static int iter_cover = 0;
    static int iter_welcome = 0;
//    abc 这个名字起的也太随意了
    static int abc = 0;
    abc++;
//    如果是内置类型的变量未被显式初始化，它的值由定义的位置所决定。定义于任何函数体之外的变量被初始化为0.一种例外情况是，定义在函数体内部的内置类型变量将不被初始化。一个未被初始化的内置类型变量的值是未定义的，如果

    if(frame_info->data == nullptr)
    {
        cerr << "Input image is empty." << endl;
        return SVIP_AI_ERR_PARAMETER;
    }

    if(frame_info->pixel_fmt != PIXEL_FMT_IYUV) return SVIP_AI_ERR_PARAMETER;
    unsigned long long t = frame_info->timestamp;
    auto ret = checkMD5(frame_info, md5_);
    cv::Mat bgr;

    auto yuv = cv::Mat(int(frame_info->height) * 3 / 2, int(frame_info->width), CV_8UC1, frame_info->data);

    if(yuv.data == nullptr)
    {
        cerr << "Create yuv error..." << endl;
        return SVIP_AI_ERR_PARAMETER;
    }

    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_IYUV);

    if(bgr.data == nullptr)
    {
        cerr << "Convert yuv error..." << endl;
        return SVIP_AI_ERR_PARAMETER;
    }
    cout << "准备送入图片预处理" << endl;
    auto img = preprocessImg(bgr, frame_info);
    cout << "图片预处理正常" << endl;

    if(img.data == nullptr)
    {
        cerr << "preprocessImg error..." << endl;
        return SVIP_AI_ERR_PARAMETER;
    }

    if (ret != SVIP_AI_OK) return ret;
    ret = checkFMT(frame_info);
    if (ret != SVIP_AI_OK) return ret;
    mytime = frame_info->timestamp;


    if (applicationType == APPLICATION_AST)
    {
        if (detector == nullptr)
        {
            cerr << "Detector was not initialized" << endl;
            return -1;
        }
        if (tracker == nullptr)
        {
            cerr << "Tracker was not initialized" << endl;
            return -1;
        }
        if (poseEst == nullptr)
        {
            cerr << "Pose Estimation was not initialized" << endl;
            return -1;
        }


        cout << "准备送入网络推理" << endl;
        ret = detector->forward(img);
        cout << "网络推理正常" << endl;
        if (ret != 0)
        {
            cerr << "forward error.. " << endl;
            return ret;
        }
        vector<DETECT::DET_BOX> detBoxes;
        parseDetRes(detector->bboxes, detBoxes);
        cout << "准备目标跟踪" << endl;
        ret = tracker->track(detBoxes);
        cout << "目标跟踪正常" << endl;
        if (ret != 0) return ret;
    }
    //// parse之前，已经经过了detection和track，也就是每一帧检测的bbox和预测出来的bbox

    if (cameraType != CAMERA_ASSISTANT_ACTION)
    {
        cerr << "Invalid camera type" << endl;
        return -1;
    }

    auto number = tracker->trkBoxes.size();
//    number代表在当前帧，一共检测到了多少个不同id
    PedAIResult aiResult;
    PedAction action[number];
    RectF bbox[number];

    int id[number];
    // 第一个数代表上一帧，第二个数代表本帧，
    pair<vector<int>, vector<int>> pair_id(999, 999);

    // 不知道在这儿加这个vector合不合适
    vector<int> CNT(100, 0);
    vector<vector<uchar>> single_ID_data;

    for (size_t i = 0; i < number; i++)
    {
        auto left = float(tracker->trkBoxes[i].xmin) / float(frame_info->width);
        auto top = float(tracker->trkBoxes[i].ymin) / float(frame_info->height);
        auto right = float(tracker->trkBoxes[i].xmax) / float(frame_info->width);
        auto bottom = float(tracker->trkBoxes[i].ymax) / float(frame_info->height);

        bbox[i].left = (left >= 0) ? left : 0;
        bbox[i].top = (top >= 0) ? top : 0;
        bbox[i].right = (right <= 1) ? right : 1;
        bbox[i].bottom = (bottom <= 1) ? bottom : 1;
//        char str[10];
//        id[i] = tracker->trkBoxes[i].ID;
//        cv::imshow("fdsaf", img);
//        cv::waitKey(0);
//        cout << bgr.size << endl;
//        cout << ymax << endl;
//        cout << frame_info->width << endl;
//        cout << xmin << ", " << ymin << ", " << xmax-xmin << ", " << ymax-ymin << endl;
//        不同的ID序列图片保存到指定文件夹下
//        cv::Mat roi(bgr,cv::Rect(xmin,ymin,xmax-xmin,ymax-ymin));

        cv::Mat img_roi = crop_util(bgr, tracker->trkBoxes[i]);
        pair<int, Mat> curr_pair = make_pair(id[i], img_roi);

//todo        pre_IDs是保存在BRIDGE这个类里的，也就是说，我不能在GetResult之前释放掉这块内存！
        poseEst->dynamic_save(pre_IDs, curr_pair);
////    结构不对，这是在BRIDGE里面，无法使用f_save操作。既然poseEst要用在这儿，

//        string picSavDir = "../sav/"+to_string(id[i])+"/";
//        sprintf(str,"%04d",CNT[id[i]] + abc);
//        cout << "准备保存至" << picSavDir << endl;
//        cv::imwrite(picSavDir + str+".jpg",img_roi);
    }
//    poseEst->dynamic_save(single_ID_data, img_roi);

//    single_ID_data.clear();
//  现在有个问题，姿态估计需要送入一个视频序列的图片，这个时候就需要解析这个内存块
//    poseEst->dynamic_load(poseEst->block_left);

//    至此得到_kps_seq，至此，送入action网络动作识别即可！！

    // 这儿的action不太对劲儿
    aiResult.ped_action = action;
    aiResult.all_target_box_rt = bbox;
    aiResult.target_box_rt_size = number;
    aiResult.id = id;
    aiResult.timestamp = frame_info->timestamp;
    aiResult.ped_ai_frame = frame_info;

    auto size = sizeof(PedAIResult)
                + aiResult.target_box_rt_size * (sizeof(PedAction) + sizeof(int) + sizeof(RectF))
                + sizeof(PedAIFrame) + frame_info->data_size * sizeof(unsigned char);
    mycallback(applicationType, cameraType, (void *) this, &aiResult, size, myuser);


    if (ret != 0) return ret;

    return 0;
}

int BRIDGE::Deinit() const
{
    auto ret = detector->Deinit();
    if (detector)
    {delete detector;}
    if (tracker)
    {delete tracker;}
    if (poseEst)
    {delete poseEst;}
//    if (actRecognizer)
//    {delete actRecognizer;}
    return ret;
}

int SVIP_AI_Ast_GetResult(ApplicationType application_type, CameraType camera_type, void *ai_params, int ai_params_size, svip_ai_result_cb cb, void *user, void **ai_handle) {
/// 这个接口相当于实现了三个功能，所以这个句柄应该要变了
//// 1. 加载模型
//// 2. 模型推理
//// 3. 返回结果

    if (APPLICATION_AST != application_type) {
        cerr << "Ast_Start: Application Type Parameter Error" << endl;
        return SVIP_AI_ERR_PARAMETER;
    }

    if (CAMERA_ASSISTANT_ACTION != camera_type) {
        cerr << "Ast_Start: Camera Type Parameter Error" << endl;
        return SVIP_AI_ERR_PARAMETER;
    }

    if (!ai_params || sizeof(AstAIParam) != ai_params_size) {
        cerr << "Ast_Start: AI Parameter Error" << endl;
        return SVIP_AI_ERR_PARAMETER;
    }

    *ai_handle = nullptr;
    //todo 这个参数的作用在这儿也就是加载路径的时候用用
    auto param = (AstAIParam *) ai_params;
    auto bridge = new ACT_BRIDGE;

    if (bridge)
        {
            string str_model_path = param->path;
            // nRet = bridge->actRecognizer->(param->path);
            string asgcn_model1 = str_model_path + "/cpp_model1.pt";
            string asgcn_model2 = str_model_path + "/cpp_model2.pt";

            if (!bridge->actRecognizer) {
                bridge->actRecognizer = new ActRecognizer;
            }
            int nRet = bridge->actRecognizer->Asgcn_model_load(asgcn_model1, asgcn_model2);
            if (nRet != 0) {
                delete bridge->actRecognizer;
                delete bridge;
                cerr << "Ast_GetResult: Init Error" << endl;
                cerr << "Ast_GetResult: recognition model Error" << endl;
                return SVIP_AI_ERROR;
            }

            if (bridge->actRecognizer == nullptr)
            {
                cerr << "Asgcn ActRecognizer was not initialized" << endl;
                return -1;
            }
//// 开始分析，首先还是送入的CSV
            string f = "../tests/subject1_file_100.csv";
            auto kps_seq = bridge->actRecognizer->Asgcn_Infer(f);
            bridge->actRecognizer->Asgcn_preprocess(kps_seq);
            cout << "load succeed.." << endl;
            nRet = bridge->actRecognizer->Asgcn_GetResult();
            if (nRet != 0) {
                delete bridge->actRecognizer;
                delete bridge;
                cerr << "Ast_GetResult: Get Results Error" << endl;
                return SVIP_AI_ERROR;
            }
            cout << "infer succeed.." << endl;
        }

    return 0;
}
