#ifndef SVIP_AI_SDK_H
#define SVIP_AI_SDK_H

// #include "opencv2/opencv.hpp"
// #include "opencv2/dnn.hpp"
// #include "opencv2/imgcodecs.hpp"
// #include "opencv2/imgproc.hpp"
// #include <opencv2/highgui.hpp>
// #include <fstream>
// #include <sstream>
// #include <iostream>
#include <stdint.h> // int64_t
#include <string>
#include <vector>

// typedef unsigned long long uint64_t;

/*
说明：
1.该.so 文件，使用时，请放置于/usr/lib
2.本次修改时间2019年11月5日，完善了参数命名，例如：老师3种行为（teach、walk、write），学生4种行为（listen、raisehands、lowerhead、stand），阈值参数confThreshold
3. 增加数据格式 PIXEL_FMT_BGRP, 3516使用
4. 增加设置图像接口 SVIP_AI_SetFrame, 用于设置分析图像格式及宽高, 后续 SVIP_AI_InputFrame 中需要判断是否是合法图像数据
5. 增加md5加密, 格式: bandsoft_ApplicationType_CameraType_timestamp, 例如: 教育类型下老师摄像头, 送入图像数据时间戳(frame_info->timestamp, 64位), bandsoft_0_0_36507222016 对应md5 7deb9a15f6ad7e98fc7461558a471a56
6. 初始化增加AI模型路径, 如果存储不够可以挂载sd卡
7. 修改6中模型路径为模型的内存地址和模型大小,可以使用服务器存储,发配模型 修改了typedef struct EduAIParam结构体作为传入途径
8. 整理代码顺序, 加入货架分析相关定义
9. 新增加一个结构体 AIFrame 表示要送入分析的数据, 各种类型下有各自的定义, 统一转为void *使用
10. 现在货架分析分为四种, 每种单独一个库, 通过函数名区分, 格式: SVIP_AI_ShelfPhoto_xxx, SVIP_AI_ShelfFront_xxx, SVIP_AI_ShelfSide_xxx, , SVIP_AI_ShelfFresh_xxx
11. ShelfAIParam结构体增加ocr_path字段
12. 增加行人统计应用相关定义, api不变
13. 封装拍照, 正拍分析结果描述
14: 增加店员行为识别应用相关定义
15: 增加V2版本斜拍
*/

#ifdef __cplusplus
extern "C" {
#endif

// 错误码定义
enum {
    SVIP_AI_OK            = 0,  // 成功
    SVIP_AI_ERROR         = -1, // 失败, 通用
    SVIP_AI_UNINITIALIZED = -2, // 未初始化
    SVIP_AI_UNSTARTED     = -3, // 未开始
    SVIP_AI_MEMORY        = -4, // 内存错误
    SVIP_AI_ERR_PARAMETER = -5, // 参数错误
    SVIP_AI_ERR_PARSER    = -6, // 解析错误
    SVIP_AI_ERR_VERIFY    = -7, // 验证错误
    /// 使用时根据需要在下方扩展
};

// 应用主类型定义
typedef enum ApplicationType {
    APPLICATION_EDU,   // 教育
    APPLICATION_BDT,   // 商业数据统计
    APPLICATION_VMS,   // 车载系统
    APPLICATION_SHELF, // 货架商品审计, 分四种类型: 拍照, 摄像头正拍, 摄像头斜拍(变化比), 生鲜标签
    APPLICATION_PED,   // 行人统计
    APPLICATION_AST,   // 店员
    /// 使用时根据需要在下方扩展
} ApplicationType;

// 摄像头类型, 应用子类型
typedef enum CameraType {
    CAMERA_EDU_TEACHER_SINGLE = 0, // 教育类型下教师摄像头, 单行为, teach
    CAMERA_EDU_STUDENT_SINGLE,     // 教育类型下学生摄像头, 单行为, stand
    CAMERA_EDU_TEACHER_MULTI,      // 教育类型下教师摄像头, 多行为, teach, write
    CAMERA_EDU_STUDENT_MULTI,      // 教育类型下学生摄像头, 多行为, listen, stand, lowerhead, raisehands, walk, person
    CAMERA_EDU_STUDENT_EXHIBITION, // 教育类型下学生摄像头, 展厅, stand
    CAMERA_SHELF_PHOTO,            // 货架类型下拍照模式
    CAMERA_SHELF_FRONT,            // 货架类型下摄像头正拍模式
    CAMERA_SHELF_SIDE,             // 货架类型下摄像头斜拍模式
    CAMERA_SHELF_FRESH,            // 货架类型下摄像头生鲜模式
    CAMERA_PEDESTRIAN_VISION,      // 行人统计类型下摄像头远景 检测人体
    CAMERA_PEDESTRIAN_CLOSE,       // 行人统计类型下摄像头近景 检测人头
    CAMERA_ASSISTANT_ACTION,       // 店员行为识别
    CAMERA_SHELF_SIDE_V2,          // 货架类型下摄像头斜拍模式, V2版本
    /// 使用时根据需要在下方扩展
} CameraType;

// 点坐标, 相对位置, 浮点型
typedef struct PointF {
    float x = 0.0f, y = 0.0f;
} PointF;

// 线坐标, 相对位置, 浮点型
typedef struct LineF {
    PointF pt1 = {}, pt2 = {};
} LineF;

// 矩形坐标, 相对位置, 浮点型
typedef struct RectF {
    float left = 0.0f, top = 0.0f, right = 0.0f, bottom = 0.0f;
} RectF;

// 图像数据
typedef enum PIXEL_FMT {
    PIXEL_FMT_YV12,  // YVU420P, Planar YVU, YYYYYYYY VV UU
    PIXEL_FMT_IYUV,  // YUV420P, Planar YUV, YYYYYYYY UU VV
    PIXEL_FMT_NV12,  // YUV420P, YYYYYYYY UV UV
    PIXEL_FMT_NV21,  // YUV420P, YYYYYYYY VU VU
    PIXEL_FMT_YUY2,  // YUYV422, YUYV YUYV
    PIXEL_FMT_UYVY,  // UYVY422, Packed UYVY, UYVY UYVY
    PIXEL_FMT_RGB24, // RGB24, Packed RGB
    PIXEL_FMT_BGR24, // BGR24, Packed BGR
    PIXEL_FMT_RGB32, // RGB32, Packed RGBA
    PIXEL_FMT_RGB15, // X1R5G5B5
    PIXEL_FMT_RGB16, // R5G6B5
    PIXEL_FMT_BGRP,  // planar BGR 4:4:4 24bpp
    PIXEL_FMT_JPEG,  //
    PIXEL_FMT_UNDEFINE = 0x7fffffff,
} PIXEL_FMT;

typedef struct FrameInfo {
    unsigned char *data;          // 数据
    unsigned int   data_size;     // 数据长度
    unsigned int   width, height; // 宽度, 高度
    PIXEL_FMT      pixel_fmt;     // 数据格式
    int64_t        timestamp;     // 时间戳

    int reserved[6]; // 保留字段
} FrameInfo;

////////////////////////////////////////////////// 教育应用相关定义 //////////////////////////////////////////////////
// 对应教育应用的分析参数
typedef struct EduAIParam {
    float confThreshold; // 设置阈值0.56, 小于该得分的框不标出
    /// ai模型内存
    unsigned char *ai_module_data;
    unsigned int   ai_module_data_size;
    /// 使用时根据需要在下方扩展
} EduAIParam;

// 对应教育应用的行为类型
typedef enum EduAction {
    EDU_ACTION_TEACHER_write,        // 教师板书
    EDU_ACTION_TEACHER_teach,        // 教师讲授
    EDU_ACTION_STUDENT_teacher_walk, // 教师巡课
    EDU_ACTION_STUDENT_stand,        // 学生起立
    EDU_ACTION_STUDENT_raisehands,   // 学生举手
    EDU_ACTION_STUDENT_lowerhead,    // 学生低头
    EDU_ACTION_STUDENT_listen,       // 学生听课
    EDU_ACTION_STUDENT_person,       // 未识别出动作的人
    /// 使用时根据需要在下方扩展
} EduAction;

typedef FrameInfo EduAIFrame;

// 对应教育应用的分析结果
typedef struct EduAIResult {
    EduAction *edu_action;
    int64_t    timestamp; // FrameInfo的时间戳

    RectF *all_target_box_rt;  // 目标框
    int    target_box_rt_size; // 目标框数目

    EduAIFrame *edu_ai_frame; // 送入分析的数据

    int reserved[6]; // 保留字段
} EduAIResult;
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////// 行人统计应用相关定义 //////////////////////////////////////////////////
// 对应行人统计应用的分析参数
typedef struct PedAIParam {
    float confThreshold; // 设置阈值0.56, 小于该得分的框不标出
    /// ai模型内存
    unsigned char *ai_module_data;
    unsigned int   ai_module_data_size;

    bool  flag; // true: 统计进入人数, false: 不统计进入人数
    LineF line; // 线
    /// 使用时根据需要在下方扩展
} PedAIParam;

// 对应行人统计应用的行为类型
typedef enum PedAction {
    PED_ACTION_PERSON_body, // 用于远景中检测人体
    PED_ACTION_PERSON_head, // 用于近景中检测人头
    /// 使用时根据需要在下方扩展
} PedAction;

typedef FrameInfo PedAIFrame;

// 对应行人统计应用的分析结果
typedef struct PedAIResult {
    PedAction *ped_action;
    int *      id;        // 每个行人的id
    int64_t    timestamp; // FrameInfo的时间戳

    RectF *all_target_box_rt;  // 目标框
    int    target_box_rt_size; // 目标框数目

    int ped_count_1; // 行人过线数量(点1->点2顺时针方向)
    int ped_count_2; // 行人过线数量(点1->点2逆时针方向)

    PedAIFrame *ped_ai_frame; // 送入分析的数据

    int reserved[6]; // 保留字段
} PedAIResult;
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////// 货架分析相关定义 //////////////////////////////////////////////////
// 对应货架应用的分析参数
typedef struct ShelfAIParam {
    float confThreshold;
    /// ai模型内存
    unsigned char *ai_module_data;
    unsigned int   ai_module_data_size;
    std::string    ocr_path; // ocr路径
    /// 使用时根据需要在下方扩展
} ShelfAIParam;

// 对应货架应用的行为类型
typedef enum ShelfAction {
    SHELF_ACTION_POSITION_WRONG, // 商品摆放错误
    SHELF_ACTION_GOODS_WRONG,    // 商品不属于本货架
    SHELF_ACTION_GOODS_MISSING,  // 缺货
    SHELF_ACTION_RESULT,         // 结果
    /// 使用时根据定义扩展
} ShelfAction;

// 对应拍照模式的分析数据
typedef struct ShelfPhotoAIFrame {
    FrameInfo * frame_info;
    std::string ini_file; // ini配置文件
} ShelfPhotoAIFrame;

// 对应摄像头正拍模式的分析数据
typedef struct ShelfFrontAIFrame {
    FrameInfo *                      frame_info;           // 从摄像头取得的图像(目前是 PIXEL_FMT_IYUV 格式)
    std::string                      ini_file;             // ini配置文件
    std::vector<std::vector<PointF>> all_analysis_polygon; // 分析区域, 不规则图形, 可以1个或多个
} ShelfFrontAIFrame;

// 对应摄像头斜拍模式的分析数据
typedef struct ShelfSideAIFrame {
    FrameInfo *                      src_frame_info;       // 从摄像头取得的图像(目前是 PIXEL_FMT_IYUV 格式)
    FrameInfo *                      frame_info;           // 从摄像头取得的图像(目前是 PIXEL_FMT_IYUV 格式)
    std::vector<std::vector<PointF>> all_analysis_polygon; // 分析区域, 不规则图形, 可以1个或多个
} ShelfSideAIFrame;

// 对应摄像头斜拍模式的分析数据, V2版本
typedef struct ShelfSideV2AIFrame {
    FrameInfo *                      src_frame_info;       // 从摄像头取得的图像(目前是 PIXEL_FMT_IYUV 格式)
    FrameInfo *                      frame_info;           // 从摄像头取得的图像(目前是 PIXEL_FMT_IYUV 格式)
    std::vector<std::vector<PointF>> all_analysis_polygon; // 分析区域, 不规则图形, 可以1个或多个
    std::string                      ini_file;             // ini配置文件
} ShelfSideV2AIFrame;

// 对应摄像头生鲜模式的分析数据
typedef struct ShelfFreshAIFrame {
    FrameInfo *                      frame_info;           // 从摄像头取得的图像(目前是 PIXEL_FMT_IYUV 格式)
    std::vector<std::vector<PointF>> all_analysis_polygon; // 分析区域, 不规则图形, 可以1个或多个
} ShelfFreshAIFrame;

typedef enum GoodsErrorType {
    GoodError_Position  = 1,   // 位置错误
    GoodError_Less_Col  = 2,   // 少排面
    GoodError_More_Col  = 4,   // 多排面
    GoodError_Miss      = 8,   // 缺货
    GoodError_Leak      = 16,  // 漏摆
    GoodError_Swing     = 32,  // 多摆
    GoodError_Less_Row  = 64,  // 少层积数
    GoodError_More_Row  = 128, // 多层积数
    GoodError_Direction = 256, // 方向错误
} GoodsErrorType;

typedef struct GoodsErrorDetail {
    std::string barcode;
    int         row, column;
    int         flag;   // 错误位, 多个GoodsErrorType &所得
    std::string remark; // 备注
} GoodsErrorDetail;

// 对应的拍照模式分析结果
typedef struct ShelfPhotoAIResult {
    ShelfAction *shelf_action;
    RectF *      all_target_box_rt;  // 目标框
    int          target_box_rt_size; // 目标框数目

    void *                        ai_frame;         // 送入分析的数据, 针对不同的类型, 对应不同的ShelfXXXAIFrame
    FrameInfo *                   ai_result_frame;  // 分析结果帧
    std::string                   ai_result_detail; // 分析结果描述
    std::vector<GoodsErrorDetail> all_goods_error_detail;

    int reserved[6]; // 保留字段
} ShelfPhotoAIResult;

typedef ShelfPhotoAIResult ShelfFrontAIResult; // 摄像头正拍模式与拍照模式分析结果相同

// 对应摄像头斜拍模式的分析结果
typedef struct ShelfSideAIResult {
    ShelfAction *shelf_action;
    RectF *      all_target_box_rt;  // 目标框
    int          target_box_rt_size; // 目标框数目
    /// TODO: 虽然用不到上面的定义但是还是先保留下来

    void *      ai_frame;         // 送入分析的数据
    FrameInfo * ai_result_frame;  // 分析结果帧, 用不到的话先为空
    std::string ai_result_detail; // 分析结果描述

    int reserved[6]; // 保留字段
} ShelfSideAIResult;

typedef ShelfPhotoAIResult ShelfSideV2AIResult; // 与拍照模式分析结果相同

typedef struct FreshDetail {
    int                 index;            // 不知道是否有用, 不用可以不赋值
    std::string         name;             // 名称
    double              price;            // 价格
    std::vector<PointF> analysis_polygon; // 分析区域, 不知道是否有用, 不用可以不赋值
} FreshDetail;

// 对应摄像头生鲜模式的分析结果
typedef struct ShelfFreshAIResult {
    ShelfAction *shelf_action;
    RectF *      all_target_box_rt;  // 目标框
    int          target_box_rt_size; // 目标框数目
    /// TODO: 虽然用不到上面的定义但是还是先保留下来

    void *                   ai_frame;         // 送入分析的数据
    FrameInfo *              ai_result_frame;  // 分析结果帧(如果需要上传图片的话需要分析库内部生成一张图片并放入图片数据)
    std::string              ai_result_detail; // 分析结果描述
    std::vector<FreshDetail> all_fresh_detail; //

    int reserved[6]; // 保留字段
} ShelfFreshAIResult;
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////// 店员行为识别应用相关定义 //////////////////////////////////////////////////
// 对应店员行为识别应用的分析参数
typedef struct AstAIParam {
    float confThreshold; // 设置阈值0.56, 小于该得分的框不标出
    float motivation;
    /// ai模型内存
    unsigned char *ai_module_data;
    unsigned int   ai_module_data_size;

    std::string path; // 路径 /// TODO: 先不用 ai_module_data, 模型文件都由库内部自己读取
    /// 使用时根据需要在下方扩展
} AstAIParam;

// 对应店员行为识别应用的行为类型
typedef enum AstAction {
    AST_ACTION_UNDEFINE,        // 未定义类型
    AST_ACTION_MOPPING,         // 刷地
    AST_ACTION_COVER_THE_CLOTH, // 叶菜湿布
    /// 使用时根据需要在下方扩展
} AstAction;

// 对应店员行为识别应用的分析数据
typedef struct AstAIFrame {
    FrameInfo *frame_info;
} AstAIFrame;

// 对应店员行为识别应用的分析结果
typedef struct AstAIResult {
    AstAction *ast_action;
    RectF *    all_target_box_rt;  // 目标框 /// TODO: 目前没有, 但是要占位, 使用 0.0-0.0-0.0-0.0 占位
    int        target_box_rt_size; // 目标框数目 /// TODO: 如果没有相关行为则为0

    void *      ai_frame;         // 送入分析的数据
    FrameInfo * ai_result_frame;  // 分析结果帧, 用不到的话先为空
    std::string ai_result_detail; // 分析结果描述

    int reserved[6]; // 保留字段
} AstAIResult;
////////////////////////////////////////////////////////////////////////////////////////////////////

/// \param application_types: 应用主类型, SVIP_AI_Start 时传入
/// \param camera_type: 应用子类型, SVIP_AI_Start 时传入
/// \param ai_handle: 分析句柄, 一路分析的标识, SVIP_AI_Start 时得到
/// \param ai_result: 分析结果, 针对不同的应用类型对应不同的结构体
/// \param ai_result_size: 分析结果内存大小
/// \param user: 用户自定义参数
/// 返回值: 0: 成功, 其它: 参考上方 enum 定义
typedef int (*svip_ai_result_cb)(ApplicationType application_type, CameraType camera_type, void *ai_handle, void *ai_result, int ai_result_size, void *user);

/// svip_ai_sdk 初始化
/// 返回值: 0: 成功, 其它: 参考上方 enum 定义
int SVIP_AI_Initialize();

/// svip_ai_sdk 销毁
/// 返回值: 0: 成功, 其它: 参考上方 enum 定义
int SVIP_AI_Uninitialize();

/// svip_ai_sdk 开始一路分析
/// \param [in] application_types: 应用主类型
/// \param [in] camera_type: 应用子类型, 用于区分摄像头
/// \param [in] ai_params: 分析参数, 根据不同的应用类型及子类型传入不同的结构体
/// \param [in] ai_params_size: 分析参数内存大小
/// \param [in] cb: 分析结果回调
/// \param [in] user: 分析结果回调自定义参数
/// \param [out] ai_handle: 成功时返回的句柄
/// 返回值: 0: 成功, 其它: 参考上方 enum 定义
int SVIP_AI_Start(ApplicationType application_type, CameraType camera_type, void *ai_params, int ai_params_size, svip_ai_result_cb cb, void *user, void **ai_handle);

/// svip_ai_sdk 停止一路分析
/// \param [in] ai_handle: 成功时返回的句柄
/// 返回值: 0: 成功, 其它: 参考上方 enum 定义
int SVIP_AI_Stop(void *ai_handle);

/// svip_ai_sdk 设置分析图像格式
/// \param [in] ai_handle // 正在分析的句柄
/// \param [in] ai_frame // 数据信息
int SVIP_AI_SetFrame(void *ai_handle, void *ai_frame);

/// svip_ai_sdk 分析一帧图像
/// \param [in] ai_handle // 正在分析的句柄
/// \param [in] md5 // md5加密
/// \param [in] ai_frame // 数据信息
int SVIP_AI_InputFrame(void *ai_handle, const char *md5, void *ai_frame);

////////////////////////////////////////////////// 货架拍照模式相关定义 //////////////////////////////////////////////////
int SVIP_AI_ShelfPhoto_Initialize();
int SVIP_AI_ShelfPhoto_Uninitialize();
int SVIP_AI_ShelfPhoto_Start(ApplicationType application_type, CameraType camera_type, void *ai_params, int ai_params_size, svip_ai_result_cb cb, void *user, void **ai_handle);
int SVIP_AI_ShelfPhoto_Stop(void *ai_handle);
int SVIP_AI_ShelfPhoto_SetFrame(void *ai_handle, void *ai_frame);
int SVIP_AI_ShelfPhoto_InputFrame(void *ai_handle, const char *md5, void *ai_frame);
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////// 货架摄像头正拍模式相关定义 //////////////////////////////////////////////////
int SVIP_AI_ShelfFront_Initialize();
int SVIP_AI_ShelfFront_Uninitialize();
int SVIP_AI_ShelfFront_Start(ApplicationType application_type, CameraType camera_type, void *ai_params, int ai_params_size, svip_ai_result_cb cb, void *user, void **ai_handle);
int SVIP_AI_ShelfFront_Stop(void *ai_handle);
int SVIP_AI_ShelfFront_SetFrame(void *ai_handle, void *ai_frame);
int SVIP_AI_ShelfFront_InputFrame(void *ai_handle, const char *md5, void *ai_frame);
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////// 货架摄像头斜拍模式相关定义 //////////////////////////////////////////////////
int SVIP_AI_ShelfSide_Initialize();
int SVIP_AI_ShelfSide_Uninitialize();
int SVIP_AI_ShelfSide_Start(ApplicationType application_type, CameraType camera_type, void *ai_params, int ai_params_size, svip_ai_result_cb cb, void *user, void **ai_handle);
int SVIP_AI_ShelfSide_Stop(void *ai_handle);
int SVIP_AI_ShelfSide_SetFrame(void *ai_handle, void *ai_frame);
int SVIP_AI_ShelfSide_InputFrame(void *ai_handle, const char *md5, void *ai_frame);
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////// 货架摄像头斜拍模式V2版本相关定义 //////////////////////////////////////////////////
int SVIP_AI_ShelfSide_V2_Initialize();
int SVIP_AI_ShelfSide_V2_Uninitialize();
int SVIP_AI_ShelfSide_V2_Start(ApplicationType application_type, CameraType camera_type, void *ai_params, int ai_params_size, svip_ai_result_cb cb, void *user, void **ai_handle);
int SVIP_AI_ShelfSide_V2_Stop(void *ai_handle);
int SVIP_AI_ShelfSide_V2_SetFrame(void *ai_handle, void *ai_frame);
int SVIP_AI_ShelfSide_V2_InputFrame(void *ai_handle, const char *md5, void *ai_frame);
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////// 货架摄像头生鲜模式相关定义 //////////////////////////////////////////////////
int SVIP_AI_ShelfFresh_Initialize();
int SVIP_AI_ShelfFresh_Uninitialize();
int SVIP_AI_ShelfFresh_Start(ApplicationType application_type, CameraType camera_type, void *ai_params, int ai_params_size, svip_ai_result_cb cb, void *user, void **ai_handle);
int SVIP_AI_ShelfFresh_Stop(void *ai_handle);
int SVIP_AI_ShelfFresh_SetFrame(void *ai_handle, void *ai_frame);
int SVIP_AI_ShelfFresh_InputFrame(void *ai_handle, const char *md5, void *ai_frame);
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////// 店员行为识别模式相关定义 //////////////////////////////////////////////////
int SVIP_AI_Ast_Initialize();
int SVIP_AI_Ast_Uninitialize();
int SVIP_AI_Ast_Start(ApplicationType application_type, CameraType camera_type, void *ai_params, int ai_params_size, svip_ai_result_cb cb, void *user, void **ai_handle);
int SVIP_AI_Ast_Stop(void *ai_handle);
int SVIP_AI_Ast_SetFrame(void *ai_handle, void *ai_frame);
int SVIP_AI_Ast_InputFrame(void *ai_handle, const char *md5, void *ai_frame);
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif

#endif
