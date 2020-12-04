//
// Created by ljh on 2020/12/1.
//

#include "pose_est/PoseEst.h"


int PoseEst::pose_init(const char *filename) {
    //  我需要写到一个类里面去，这样相互调用就方便多了
    _model = tflite::FlatBufferModel::BuildFromFile(filename);
    InterpreterBuilder(*_model, _resolver)(&interpreter);
    interpreter->AllocateTensors();

    return 0;
}

int PoseEst::pose_deal_frame(vector<Mat> img_stream) {
//  每进一次这个函数，都会重新给_kps_seq赋值
    if (_kps_seq.data() != nullptr)
    {
        _kps_seq.clear();
    }

    Mat im_resize(257,257,CV_8UC3);
    // Build the interpreter
    for (auto img : img_stream)
    {

        cv::resize(img, im_resize, Size(257,257));
        int input_tensor_idx = 0;
        int input = interpreter->inputs()[input_tensor_idx];
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

        for (int i = 0; i < 9; i++)
            for (int j = 0; j < 9; j++)
                for (int k = 0; k < 34; k++)
                {
                    offsets[i][j][k] = *offset_data_ptr;
                    offset_data_ptr++;
                }
        int joint_num = 17;
        int pose_kps[17][2];
        float joint_heatmap[9][9];

        int max_val_pos[2];
        float max = -99;
        vector<int> single_point;
        // 处理函数
        for (int k = 0; k < 17; k++)
        {
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
//            single_point存放的是一对儿坐标点
            single_point.emplace_back(pose_kps[k][0]);
            single_point.emplace_back(pose_kps[k][1]);
            _kps.emplace_back(single_point);
            single_point.clear();
        }
        // 进行多少次emplace_back取决于for循环的次数
        _kps_seq.emplace_back(_kps);
        _kps.clear();
    }

    return 0;
}

//int PoseEst::dynamic_save(vector<vector<uchar>> &single_ID_data, cv::Mat single_frame) {
//    vector<uchar> jpg;
//    cv::imencode(".jpg", single_frame, jpg);
//    single_ID_data.emplace_back(jpg);
//    return 0;
//}

int PoseEst::dynamic_save(vector<int> &pre_IDs, pair<int, Mat> &curr_pair) {
    auto ret = is_element_in_vector(pre_IDs, curr_pair.first);

    if (ret == 1)
    {
        _index = find_index(pre_IDs, curr_pair.first);
        cout << "之前已经存在过该ID，该ID为："<< _index << endl;
        // 然后找到block_left当中对应那一块，将其送入
        block_left[_index].emplace_back(curr_pair);
    } else{
        cout << "需要继续增加prev_block，emplace_back" << endl;
        // curr_block只存放新出现的block
        curr_block.emplace_back(curr_pair);
        pre_IDs.emplace_back(curr_pair.first);
        block_left.emplace_back(curr_block);
        curr_block.clear();
    }
    return 0;
}

bool PoseEst::is_element_in_vector(vector<int> v, int element) {
    vector<int>::iterator it;
    it=find(v.begin(),v.end(),element);
    if (it!=v.end()){
        return true;
    }
    else{
        return false;
    }
}

int PoseEst::find_index(vector<int> v, int element) {
    vector<int>::iterator iElement = find(v.begin(), v.end(), element);
    int index = distance(begin(v), iElement);
    return index;
}

int PoseEst::dynamic_load(vector<vector<pair<int, Mat>>> block_left) {
    for (int i = 0; i < block_left.size(); i++)
    {
        vector<pair<int, Mat>> IDi;
        IDi = block_left[i];
        // 然后遍历所有的，把它们弄出来
        for (auto pic : IDi)
        {
            _pic_candidate.emplace_back(pic.second);
        }
//      送入pose estimation网络推理得到坐标
        pose_deal_frame(_pic_candidate);
        _pic_candidate.clear();
    }
    return 0;
}
