//
// Created by djf on 2020/10/16.
//
// todo
//  2, 设计一个缓冲区，用于存放一个视频序列的关节点坐标，当ID切换之后，满足1，缓冲区清零，此时送入动作识别网络分析
//  3, 目前送入网络的是单帧图片循环分析，这没关系，后期再提高图片帧率
//  4, 提高yolov5检测精度

#include <iostream>
#include <dirent.h>
#include "detect.h"
#include "track.h"
#include "svipAISDK_V3.h"
#include "xskutils.hpp"
#include <dirent.h>
#include <fstream>
#include <cstdio>
#include <utils/utils.h>
#include "md5/md5.h"
#include "PoseEst.h"

using namespace std;

int DetTrkPos_callback(ApplicationType application_type,CameraType camera_type, void *ai_handle, void *ai_result, int ai_result_size, void *user)
{
    return 0;
}

int result_callback(ApplicationType application_type,CameraType camera_type, void *ai_handle, void *ai_result, int ai_result_size, void *user)
{
//    auto result = (PedAIResult *) ai_result;
//    auto img = cv::imread(srcDir + imgFile);
//    auto width = img.cols;
//    auto height = img.rows;
//    vector<int> cmp;
//    vector<string> folderlist;
//    string folderpath("../sav/");
//    xskUtils::GetFileNames(folderpath,folderlist);
//
//    if(result->target_box_rt_size!=0)
//    {
//        for(size_t i=folderlist.size();i<=result->id[(result->target_box_rt_size)-1]; i++)
//            {
//                string save_folderpath("../sav/"+to_string(i));
//                string command ("mkdir "+save_folderpath);
//                system(command.c_str());
//            }
//    }
//    for (size_t i = 0; i < result->target_box_rt_size; i++)
//    {
//        auto ID = result->id[i];
//        char str[10];
//        cnt[ID]++;
//        auto xmin = int(result->all_target_box_rt[i].left * width);
//        auto xmax = int(result->all_target_box_rt[i].right * width);
//        auto ymin = int(result->all_target_box_rt[i].top * height);
//        auto ymax = int(result->all_target_box_rt[i].bottom * height);
//        cv::Mat roi(img,cv::Rect(xmin,ymin,xmax-xmin,ymax-ymin));
//        string picSavDir = "../sav/"+to_string(ID)+"/";
//        sprintf(str,"%04d",cnt[ID]);
//        cv::imwrite(picSavDir + str+".jpg",roi);
//    }
//    cnt1 += result->ped_count_1;
//    cnt2 += result->ped_count_2;
    return 0;
}

//todo: 首先要修改掉这个main函数，改成标准的加载方式：do while(false)
// 然后，之前的SVIP_AI_Ast_Stop的操作是，程序循环不断的执行，若非遇到特殊情况不会触发，也就是不会有累积的内存
// 但是现在，我需要人为的stop来清空资源，然后重新加载！！！具体来说，就是300帧一到，就先stop，然后重新start
// start的操作同样可以包含yolo，track，pose_est操作，这没问题，但是这个时候的start，还要同时启动动作识别分析。
// 放佛是伪并行的机制，分析一直在进行，但是识别在空档，但是，但是，但是，识别的时候是将内存的数据读出，这个时候就不能再往内存里面送数据了，也就是说，这段的视频数据我就丢失了。
// 那这样的话，送入的buff和读取的buff要是两个不同的buff，当启动动作分析的时候，首先要做的不是直接分析，而是将保存的buff送入到读取的buff里面去，为了不影响清空保存的buff，然后保存的buff继续保存新的数据，读取的buff就直接得到分析结果就可以了


int main()
{
    int count = 0;
    /// STEP0 模型指针获取
    MD5_CTX _md5;
    int ret = 0;
    void * ai_handle;
    // 工程规定，这儿要写半个路径，然后后面字符串拼接
    string modelPath("/home/ljh/Documents/action_detect/Final/remote_test/svip_action_Final_v2/yolov5-sort/model/best.engine");
    string picSrcDir("../data/pic/supermarkets");

//    srcDir = picSrcDir;

    ifstream file(modelPath, ios::binary);
    char *modelPtr;
    size_t FileSize;
    if (file.good())
    {
        file.seekg(0, ifstream::end);
        FileSize = file.tellg();
        file.seekg(0, ifstream::beg);
        modelPtr = new char[FileSize];
        file.read(modelPtr, FileSize);
        file.close();
    }
    else
    {
        cerr << "[ERROR] Failed to load model " << modelPath << endl;
        return -1;
    }

    SVIP_AI_Ast_Initialize();
    /// step1  初始化工程参数
    AstAIParam astAiParam;
    astAiParam.confThreshold = 0.5;  // detection 目标框阈值

    astAiParam.path = "/home/ljh/Documents/action_detect/Final/remote_test/svip_action_Final_v2/yolov5-sort/model";
    astAiParam.ai_module_data = reinterpret_cast<unsigned char *>(modelPtr);
    astAiParam.ai_module_data_size = FileSize;

    /// STEP2   将模型参数，callback，送入句柄并启动分析

    ret = SVIP_AI_Ast_Start(APPLICATION_AST, CAMERA_ASSISTANT_ACTION,
                                 &astAiParam, sizeof(astAiParam), DetTrkPos_callback, nullptr, &ai_handle);
    delete[] modelPtr;
    if (ret != 0)
    {
        cout << "SVIP_AI_Start ret = " << ret << endl;
        SVIP_AI_Ast_Stop(ai_handle);
        return ret;
    }

    FrameInfo frameInfo;
    frameInfo.timestamp = 36507222016;

    vector<string> picList;
    ljh::GetFileNames(picSrcDir, picList);

    cout << picSrcDir << endl;
    sort(picList.begin(), picList.end());
    for (auto &picFile : picList) {
        count++;
        /// step2   预加载图片
        cout << picSrcDir + '/' + picFile << endl;
        cv::Mat img_ori = cv::imread(picSrcDir + '/' + picFile);
        cv::Mat yuv;
        cv::cvtColor(img_ori, yuv, cv::COLOR_BGR2YUV_I420);
        frameInfo.data = (unsigned char *) yuv.data;
        frameInfo.data_size = img_ori.cols * img_ori.rows * 3 / 2;
        frameInfo.height = img_ori.rows;
        frameInfo.width = img_ori.cols;
        frameInfo.pixel_fmt = PIXEL_FMT_IYUV;
        cout << "图片加载正确" << endl;
        /// step3   加密相关
        char encrypt[50] = {};
        char decrypt[50] = {};
        sprintf(encrypt, "bandsoft_%d_%d_%ld", APPLICATION_AST, CAMERA_ASSISTANT_ACTION, frameInfo.timestamp);
        MD5Init(&_md5);
        MD5Update(&_md5, reinterpret_cast<unsigned char *>(encrypt), strlen(encrypt));
        MD5Final(&_md5, reinterpret_cast<unsigned char *>(decrypt));
        char md5_out[33] = {0};
        int i;
        for (i = 0; i < 16; i++) {
            sprintf(md5_out + i + i, "%02x", decrypt[i]);
        }
        /// step4   初始化
        AstAIFrame ast_ai_frame = {};
        ast_ai_frame.frame_info = &frameInfo;
        SVIP_AI_Ast_SetFrame(ai_handle, &ast_ai_frame);
        cout << "图片初始化正确" << endl;
        ret = SVIP_AI_Ast_InputFrame(ai_handle, md5_out, (void *) &ast_ai_frame);
        cout << "图片送入正确" << endl;
        if (ret != 0) {
            SVIP_AI_Ast_Stop(ai_handle);
            return ret;
        }
        //  一直在往block里面存数据
        if (count == 300)
        {
            SVIP_AI_Ast_GetResult(APPLICATION_AST, CAMERA_ASSISTANT_ACTION,
                                  &astAiParam, sizeof(astAiParam), result_callback, nullptr, &ai_handle);
        }
    }

    SVIP_AI_Ast_Uninitialize();

//    string csv_name_pre("../csvs/");
////    string label_name("../labels/");
//    string csv_name("");
//    string folder = "../sav/";
//    vector<string> foldernames;
//    xskUtils::GetFileNames(folder, foldernames);
//
//    for (int j = 0; j < foldernames.size(); j++){
//
//        vector<string> filenames;
//        xskUtils::GetFileNames(folder + foldernames[j], filenames);
//        cout << "folder_name: " << foldernames[j] << endl;
//        sort(filenames.begin(), filenames.end());
//
//        // 开始某一个csv
//        csv_name = csv_name_pre + foldernames[j] + ".csv";
//        ofstream res(csv_name);
//        res << "X1,"<<"Y1,"<<"X2,"<<"Y2,"<<"X3,"<<"Y3,"<<"X4,"<<"Y4,"<<"X5,"<<"Y5,"<<"X6,"<<"Y6,"<<"X7,"<<"Y7,"<<"X8,"<<"Y8,"<<"X9,"<<"Y9,"<<"X10,"<<"Y10,"<<"X11,"<<"Y11,"<<"X12,"<<"Y12,"<<"X13,"<<"Y13,"<<"X14,"<<"Y14,"<<"X15,"<<"Y15,"<<"X16,"<<"Y16,"<<"X17,"<<"Y17" << endl;
//        for (int i = 0; i < filenames.size(); i++){
//            Mat img;
//            char* tfliteModel = "../data/models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite";
//            cout << "image_name: " << folder + foldernames[j] + "/" + filenames[i] << endl;
//            img = cv::imread(folder + foldernames[j] + "/" + filenames[i]);
//            auto extractor = new Extract(img, tfliteModel);
//            for (int k = 0; k < extractor->_res.size(); k++){
//                res << extractor->_res[k] << "," ;
//            }
//            res << endl;
//        }
//        // 关闭这个csv
//        res.close();
//    }

    return 0;
}

