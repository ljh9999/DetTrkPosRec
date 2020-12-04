//
// Created by ljh on 2020/10/27.
//

#include <svip/svipAISDK_V3.h>
#include "crop_util.h"

/**
 * 抠图功能完成：
 * 提取出ROI之后，将其resize & padding到257*257大小之后返回
 **/

Mat crop_util(cv::Mat input, TRACK::TRK_BOX bbox)
{
//    cout << bbox.xmin << " " <<  bbox.ymin << " " << bbox.xmax << " " << bbox.ymax << endl;

    auto xmin = (float(bbox.xmin) >= 1   ) ? float(bbox.xmin) : 1;;
    auto xmax = (float(bbox.xmax) <= 1920) ? float(bbox.xmax) : 1920;
    auto ymin = (float(bbox.ymin) >= 1   ) ? float(bbox.ymin) : 1;;
    auto ymax = (float(bbox.ymax) <= 1080) ? float(bbox.ymax) : 1080;

    Rect area(xmin, ymin, xmax-xmin, ymax-ymin);
    Mat imgROI = input(area);
    /**
     * padding
     * **/
    float _rw,_rh,_re_w,_re_h,_re_x,_re_y;
    _rw = 257 / float(bbox.xmax - bbox.xmin);
    _rh = 257 / float(bbox.ymax - bbox.ymin);
    if (_rh > _rw) // original image is a horizontal rectangle
    {
        _re_w = 257;
        _re_h = int(_rw * float(bbox.ymax - bbox.ymin));
        _re_x = 0;
        _re_y = (257 - _re_h) / 2;
    }
    else // original image is a vertical rectangle
    {
        _re_w = int(_rh * float(bbox.xmax - bbox.xmin));
        _re_h = 257;
        _re_x = (257 - _re_w) / 2;
        _re_y = 0;
//        cout << "_re_w: " << _re_w << "_re_h" << _re_h << endl;
    }
    cv::Mat re(_re_h, _re_w, CV_8UC3);
    cv::resize(imgROI, re, re.size(), 0, 0, cv::INTER_LINEAR);
//    cout << "re.size(): " << re.size() << endl;

    cv::Mat out(257, 257, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(_re_x, _re_y, re.cols, re.rows)));

    return out;
}

