//
// Created by ljh on 2020/10/4.
//

#include "Extract.h"
Extract::Extract(Mat im_input, const char *filename) {
    _res.clear();
    Mat im_resize(257,257,CV_8UC3);
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(filename);
    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<Interpreter> interpreter;
    InterpreterBuilder(*model, resolver)(&interpreter);
    cv::resize(im_input, im_resize, Size(257,257));
//    cv::imshow("strange", im_resize);
//    cv::waitKey(0);
//    Resize input tensors, if desired.
// AllocateTensors作用是给那些没有内存块的张量分配内存
    interpreter->AllocateTensors();
    int input_tensor_idx = 0;
//  interpreter->inputs()[0]得到输入张量数组中的第一个张量，也就是分类器中唯一的那个输入张量。input是个整型值，语义是张量列表中的索引
    int input = interpreter->inputs()[input_tensor_idx];
//  以input为索引，在TfLiteTensor* content_.tensors这个张量表得到具体的张量。返回该张量的data.raw，它指示张量正关联着的内存块。
    float* input_data_ptr = interpreter->typed_tensor<float>(input);

    for(int i = 0; i < 257; i++)
        for(int j = 0; j < 257; j++)
            for(int k = 0; k < 3; k++)
            {
                *(input_data_ptr) = (float)(im_resize.at<Vec3b>(i,j)[k] - 127.5)/127.5;
                input_data_ptr++;
            }
    interpreter->Invoke();

    int output_tensor_idx = 0;
    int offset_tensor_idx = 1;
    int output = interpreter->outputs()[output_tensor_idx];
    int offset = interpreter->outputs()[offset_tensor_idx];

//    cout << "**************************************************************" << endl;
//    cout << interpreter->outputs()[1] << endl;
//    cout << "**************************************************************" << endl;

    float* output_data_ptr = interpreter->typed_tensor<float>(output);
    float* offset_data_ptr = interpreter->typed_tensor<float>(offset);

    float heatmaps[9][9][17];
    float offsets[9][9][34];


    for (int i = 0; i < 9; i++)
        for (int j = 0; j < 9; j++)
            for(int k = 0; k < 17; k++)
            {
                heatmaps[i][j][k] = *output_data_ptr;
                output_data_ptr++;
            }

//    cout << "heatmaps[1][0][]: " << heatmaps[1][0][0] << endl;

    for (int i = 0; i < 9; i++)
        for (int j = 0; j < 9; j++)
            for (int k = 0; k < 34; k++)
             {
                offsets[i][j][k] = *offset_data_ptr;
                offset_data_ptr++;
             }
//    cout << "offsets[][][]: " << offsets[1][0][0] << endl;
    int joint_num = 17;
    int pose_kps[17][2];
    float joint_heatmap[9][9];

    int max_val_pos[2];
    float max = -99;
    // 处理函数
    for (int k = 0; k < 17; k++)
    {
//         joint_heatmap = heatmap_data[..., i]
        for (int i = 0; i < 9; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    joint_heatmap[i][j] = heatmaps[i][j][k];
                    if (joint_heatmap[i][j] > max)
                    {
                        max_val_pos[0] = i;
                        max_val_pos[1] = j;
                        max = joint_heatmap[i][j];
                    }
                }
            }
//        cout << "max_val_pos: " << max_val_pos[0] << " " <<  max_val_pos[1] << " " << joint_heatmap[max_val_pos[0]][max_val_pos[1]] << endl;
        pose_kps[k][0] = int(max_val_pos[0]/8.*257 + offsets[max_val_pos[0]][max_val_pos[1]][k]);
        pose_kps[k][1] = int(max_val_pos[1]/8.*257 + offsets[max_val_pos[0]][max_val_pos[1]][k + joint_num]);
        max = -99;
        _res.push_back(pose_kps[k][0]);
        _res.push_back(pose_kps[k][1]);
    }

//    for (auto &i :_res)
//    {
//        cout << i << "," ;
//    }
//    cout << endl;
}

/*Extract::Extract(Mat im_input, const char *filename) {
    _res.clear();
    Mat im_resize;
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(filename);
    // Build the interpreter

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<Interpreter> interpreter;
    InterpreterBuilder(*model, resolver)(&interpreter);
    cv::resize(im_input, im_resize, Size(257,257), 3);

    cv::imshow("strange", im_resize);
    cv::waitKey(0);
    // Resize input tensors, if desired.
    interpreter->AllocateTensors();
    int input_tensor_idx = 0;
    int input = interpreter->inputs()[input_tensor_idx];

    float* input_data_ptr = interpreter->typed_tensor<float>(input);

    for(int i = 0; i < 257; ++i)
        for(int j = 0; j < 257; ++j)
            for(int k = 0; k < 3; ++k)
            {
                *(input_data_ptr) = (float)im_resize.at<Vec3b>(i,j)[k];
                input_data_ptr++;
            }
    interpreter->Invoke();

    int output_tensor_idx = 0;
    int output = interpreter->outputs()[output_tensor_idx];
    float* output_data_ptr = interpreter->typed_tensor<float>(output);
    float output_resize[1][9*9][17];
    for(int i = 0; i < 9*9; i++)
        for (int j = 0; j < 17; j++) {
            output_resize[0][i][j] = *output_data_ptr;
//            cout << output_resize[0][i][j] << endl;
            output_data_ptr++;
        }
    float max = -99.;
    long max_index[17];
    for(int i = 0; i < 17; i++) {
        for (int j = 0; j < 9 * 9; j++) {
            if (output_resize[0][j][i] > max){
                max_index[i] = j;
                max = output_resize[0][j][i];
            }
        }
        max = -99.;
        cout << "hit" << endl;
        cout << max_index[i] << endl;
    }

    for (int i = 0; i < 17; i++){
        _res.push_back(int(max_index[i]/9));
        _res.push_back(max_index[i] % 9);
    }
    for (auto &i :_res)
    {
        cout << i << "," ;
    }
    cout << endl;

}*/

































