//
// Created by ljh on 2020/12/3.
//

#include "ActRecognize.h"

int ActRecognizer::Asgcn_model_load(const string model1, const string model2) {

    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module2 = torch::jit::load("../cpp_model2.pt");
        module2.to(at::kCUDA);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module1 = torch::jit::load("../cpp_model1.pt");
        module1.to(at::kCUDA);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    return 0;
}

vector<vector<vector<int>>> ActRecognizer::Asgcn_Infer(const string f) {
    //// step1 加载原始数据
    //// todo 修改成输入多个csv

    vector<vector<vector<int>>> kps_seq = read_csv(f);

    return kps_seq;
}

int ActRecognizer::Asgcn_GetResult() {

    //    step 3-2 送入网络推理
    data_downsample = data_downsample.toType(torch::kFloat);
    data_downsample = data_downsample.to(torch::kCUDA);

    if(!data_downsample.is_cuda())
    {
        cerr << " [ERROR] Tensors are not in CUDA " << endl;
        return -1;
    }


    cout << "data_downsample: " << data_downsample.sizes() << endl;

    auto result_model2 = module2.forward({data_downsample}).toTuple();
    A_batch = result_model2->elements()[0].toTensor();



    if(!data.is_cuda() or !target_data.is_cuda() or !data_last.is_cuda() or !A_batch.is_cuda())
    {
        cerr << " [ERROR] Tensors are not in CUDA " << endl;
        return -1;
    }


    inputs.emplace_back(data);          // (1, 3, 290, 17,1)
    inputs.emplace_back(target_data);   // (1, 3, 10,  17,1)
    inputs.emplace_back(data_last);     // (1, 3, 1,   17,1)
    inputs.emplace_back(A_batch);       // (1, 2, 17,  17)

    cout << data.sizes() << endl;
    cout << target_data.sizes() << endl;
    cout << data_last.sizes() << endl;
    cout << A_batch.sizes() << endl;
    cout << inputs.size() << endl;

    //////    step 4-2 送入网络推理
    auto result_model1 = module1.forward(inputs).toTuple();
//////    step 4-3 得到网络结果
    torch::Tensor x_class = result_model1->elements()[0].toTensor();

    cout << " x_class: "<<x_class << endl;

    return 0;
}

int ActRecognizer::Asgcn_preprocess(vector<vector<vector<int>>> kps_seq) {

    //  输入维度不确定，根据视频来定
//  输入维度：input_seq shape     (512, 17 , 2) 300+代表帧数，17代表关节点，2代表x和y
//  过渡维度，得到：tmp_seq        (1,   3,   300, 17)
//  输出维度：

//  注意：data_downsample也需要传出去

    int ret;
    Tensor tmp_seq;
    auto input_seq = read_skeleton_filter(kps_seq);  //(256,17,3)



    ret = parse_kps(input_seq, tmp_seq);

    if (ret != 0){
        cerr << "parse_kps error!" << endl;
    }

    if (ret != 0){
        cerr << "dnsp error!" << endl;
    }
    // tmp_seq ——> data         (1, 3, 290, 17)
    //         ——> target_data  (1, 3, 10, 17)
    //         ——> data_last    (1, 3, 1, 17)
    //          最后得到out_seq  (1, 3, 301, 17)

    ret = split_compose(tmp_seq, data, data_downsample, target_data, data_last);
    if (ret != 0){
        cerr << "split_compose error!" << endl;
    }


    return 0;
}

int ActRecognizer::split_compose(Tensor tmp_seq, Tensor &input_data, Tensor &input_data_dnsp, Tensor &target_data,
                                Tensor &data_last) {

    //  输入维度：tmp_seq                    (1, 3, 300, 17, 1)
//  输出维度：input_data                 (1, 3, 290, 17, 1)
//  输出维度：data_downsample            (1, 3, 50,  17)
//  输出维度：target_data                (1, 3, 10, 17, 1)
//  输出维度：data_last                  (1, 3, 1,   17, 1)

//  step1: tmp_seq 分割
    vector<int> input_data_indx;
    vector<int> target_data_indx;

    for (int i = 0; i < 290; i++)
    {
        input_data_indx.push_back(i);
    }

    for (int i = 291; i < 300; i++)
    {
        target_data_indx.push_back(i);
    }

    input_data = tmp_seq.slice(2, 0, 1);

    for (int i=1; i<input_data_indx.size(); i++)
    {
        torch::Tensor append = tmp_seq.slice(2, input_data_indx[i], input_data_indx[i]+1);
        input_data = torch::cat({input_data, append}, 2);
    }

//    todo 检查input_data

    data_last = tmp_seq.slice(2, 290, 291);

    target_data = tmp_seq.slice(2, 290, 291);

    for (int i=0; i< target_data_indx.size(); i++)
    {
        torch::Tensor append = tmp_seq.slice(2, target_data_indx[i], target_data_indx[i]+1);
        target_data = torch::cat({target_data, append}, 2);
    }
//    todo 检查target_data
    cout<<target_data.sizes()<<endl;
// step1:查找有效帧，因为之前的操作有补零
    auto tmp_seq_ = tmp_seq.squeeze(0);
    auto valid_frame = ((tmp_seq_ != 0).sum(3).sum(2).sum(0)) > 0;

    cout << "valid_frame: " << valid_frame.sizes() << endl;

// step2:downsample
    auto zero_check = torch::nonzero(valid_frame);
    auto length = zero_check[-1] - zero_check[0] + 1;
    vector<int>   indx;
    for (int i = 0 ; i < 50; i++)
    {
        auto jj = int(i * (length.item().toInt() - 10) / 50);
        indx.push_back(jj);
    }

    input_data_dnsp     = input_data.slice(2, 0, 1);
    cout << "input_data_dnsp: " << input_data_dnsp.sizes() << endl;


    for (int i=1; i<indx.size(); i++)
    {
        torch::Tensor d   = input_data.slice(2, indx[i], indx[i]+1);
        input_data_dnsp = torch::cat({input_data_dnsp, d}, 2);
    }

//  step3: 送入cuda
    input_data = input_data.toType(torch::kFloat);
    input_data = input_data.to(torch::kCUDA);

    input_data_dnsp = input_data_dnsp.toType(torch::kFloat);
    input_data_dnsp = input_data_dnsp.to(torch::kCUDA);

    target_data = target_data.toType(torch::kFloat);
    target_data = target_data.to(torch::kCUDA);

    data_last = data_last.toType(torch::kFloat);
    data_last = data_last.to(torch::kCUDA);

    return 0;
}

int ActRecognizer::padding_zero(Tensor tmp_tmp_seq, Tensor &fp) {

    //  输入维度：tmp_tmp_seq：(1, 3, 92, 17)
//  输出维度：fp  (1, 3, 300, 17)
//  注意维度的变换
//  fp = np.zeros((1, 3, max_frame, num_joint, max_body_true), dtype=np.float32)
//  fp[0, :, 0:data.shape[1], :, :] = data
//    fp = torch::zeros({1,3,300,17});
    int shape3 = tmp_tmp_seq.size(2);
    torch::Tensor zero_pad = torch::zeros({1, 3, 300-shape3, 17});

    fp = torch::cat({tmp_tmp_seq, zero_pad}, 2);
    fp = fp.unsqueeze(-1);

    return 0;
}

int ActRecognizer::parse_kps(vector<vector<vector<int>>> input_seq, Tensor &tmp_seq) {

    //  输入维度：input_seq：        (512, 17 , 3)       30x代表帧数，17代表关节点，2代表x和y
//  中间维度：tmp_tmp_seq：      (3,   256, 17,  1)
//  中间维度：fp                 (1,   3,   300, 17, 1)
//  输出维度：tmp_seq：          (1,   3,   300, 17, 1) 1代表batch，3代表xyz，300代表300帧，不足300帧的用0填充
    int ret;
    Tensor tmp_tmp_seq;
    Tensor fp;

//  输出a 维度：  (512, 17, 3) type vector
//    auto a = read_skeleton_filter(intput_seq);

    tmp_tmp_seq = read_xyz(input_seq);
    // todo read_xyz的输出有问题！！

//    cout << "out" << endl;
//    cout << tmp_tmp_seq.sizes() << endl;
//
//    string debug_ = "debug.txt";
//    ofstream debug(debug_);
//    debug << tmp_tmp_seq << endl;

    ret = padding_zero(tmp_tmp_seq, fp);

    tmp_seq = pre_normalization(fp);

    if (ret != 0){
        cerr << "pre_normalization error..." << endl;
        return -1;
    }

    return 0;
}

std::vector<int> ActRecognizer::split(std::string str, std::string pattern) {
    std::string::size_type pos;
    std::vector<int> result;
    str += pattern;//扩展字符串以方便操作
    int size = str.size();
    for (int i = 0; i < size; i++) {
        pos = str.find(pattern, i);
        if (pos < size) {
            std::string s = str.substr(i, pos - i);
            int s_int = stoi(s);
            result.push_back(s_int);
            i = pos + pattern.size() - 1;
        }
    }
    return result;
}

std::vector<vector<int>> ActRecognizer::line2coor(vector<int> a) {
    if (a.size() % 2) {
        throw a.size();
    }
    vector<vector<int>> single_kps(a.size() / 2);
    for (int i = 0; i < a.size() / 2; i++) {
        single_kps[i].push_back(a[2 * i]);
        single_kps[i].push_back(a[2 * i + 1]);
    }

    return single_kps;
}

vector<vector<vector<int>>> ActRecognizer::read_csv(string filepath) {
    vector<vector<vector<int>>> kps_seq;
    ifstream csv(filepath);
    string line;
    if (csv) {
        while (getline(csv, line)) {
            line.erase(line.find_last_not_of(' ') + 1, string::npos);
            string f = " ";
            vector<int> res = split(line, f);
//            cout << res << endl;
            kps_seq.push_back(line2coor(res));
        }
    } else {
        cout << "False";
    }
    return kps_seq;
}

vector<vector<vector<int>>> ActRecognizer::read_skeleton_filter(vector<vector<vector<int>>> kps_seq) {
    // 输入维度：  (512, 17, 2)
    // 输出维度：  (256, 17, 3)

    vector<vector<vector<int>>> data;
    // 跳帧gap
    int gap = kps_seq.size() / 300 + 1;
    for (int i = 0; i < kps_seq.size() / gap; i++) {
        for (int j = 0; j < kps_seq[i * gap].size(); j++) {
            kps_seq[i * gap][j].push_back(0);
        }
        data.push_back(kps_seq[i * gap]);
    }
    return data;
}

//int ActRecognizer::read_xyz(vector<vector<vector<int>>> s, Tensor &tmp_tmp_seq) {

Tensor ActRecognizer::read_xyz(vector<vector<vector<int>>> s) {
    // 输入维度：(92, 17 , 3) type vector
    // 输出维度：(1, 3, 92, 17)
    auto ss = linearize(s);
    torch::Tensor t = torch::from_blob(ss.data(), {int(s.size()), int(s[0].size()),  int(s[0][0].size())});

    torch::Tensor tt = t.unsqueeze(0);

    auto tmp_tmp_seq = tt.permute({0, 3, 1, 2}).contiguous();
//    cout << "in" << endl;
//    cout << tmp_tmp_seq.sizes() << endl;
//
//    string debug_ = "debug1.txt";
//    ofstream debug(debug_);
//    debug << tmp_tmp_seq << endl;

    return tmp_tmp_seq;
}

torch::Tensor ActRecognizer::angle_between(torch::Tensor v1, torch::Tensor v2) {

    if (torch::abs(v1).sum().item().toDouble() < 0.000001 || torch::abs(v2).sum().item().toDouble() < 0.000001){
        return torch::zeros({1});
    }

    torch::Tensor v1_u = v1 / v1.norm();
    torch::Tensor v2_u = v2 / v2.norm();
    return torch::arccos(torch::clip(torch::dot(v1_u, v2_u), -1.0, 1.0));

    return torch::Tensor();
}

torch::Tensor ActRecognizer::rotation_matrix(torch::Tensor axis, torch::Tensor theta) {

    if (torch::abs(axis).sum().item().toFloat() < 0.000001 || torch::abs(theta).sum().item().toFloat() < 0.000001)
        return torch::eye(3);
    axis = axis / torch::sqrt(torch::dot(axis, axis));
    torch::Tensor a = torch::cos(theta / 2.0);
    torch::Tensor tmp = -axis * torch::sin(theta / 2.0);
    torch::Tensor idx = torch::empty({1}).toType(torch::kLong);
    idx[0] = 0;
    torch::Tensor b = tmp.index_select(0, idx);
    idx[0] = 1;
    torch::Tensor c = tmp.index_select(0, idx);
    idx[0] = 2;
    torch::Tensor d = tmp.index_select(0, idx);
    torch::Tensor aa = (a * a);
    torch::Tensor bb = b * b;
    torch::Tensor cc = c * c;
    torch::Tensor dd = d * d;
    torch::Tensor bc = b * c;
    torch::Tensor ad = a * d;
    torch::Tensor ac = a * c;
    torch::Tensor ab = a * b;
    torch::Tensor bd = b * d;
    torch::Tensor cd = c * d;
    torch::Tensor ret = torch::empty({3, 3});
    ret[0] = torch::cat({aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)}, 0);
    ret[1] = torch::cat({2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)}, 0);
    ret[2] = torch::cat({2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc}, 0);
    return ret;
}

torch::Tensor ActRecognizer::pre_normalization(torch::Tensor data) {
    int zaxis[] = {0, 1};
    int xaxis[] = {8, 4};

    torch::Tensor s = data.permute({0, 4, 2, 3, 1});

    for (int i_s = 0; i_s < s.sizes()[0]; i_s++){
        torch::Tensor skeleton = s[i_s];
        if (!skeleton.sum().item().toBool())
            continue;
        torch::Tensor idx = torch::empty({1}).toType(torch::kLong);
        idx[0] = 1;
        torch::Tensor main_body_center = skeleton[0].index_select(1, idx);
        for (int i_p = 0; i_p < skeleton.sizes()[0]; i_p++){
            torch::Tensor person = skeleton[i_p];
            if (!person.sum().item().toBool())
                continue;
            torch::Tensor mask = (person.sum(-1) != 0).reshape({s.sizes()[2], s.sizes()[3], 1});
            s[i_s][i_p] = (s[i_s][i_p] - main_body_center) * mask;
        }
    }


    for (int i_s = 0; i_s < s.sizes()[0]; i_s++){
        torch::Tensor skeleton = s[i_s];
        if (!skeleton.sum().item().toBool())
            continue;
        torch::Tensor joint_bottom = skeleton[0][0][zaxis[0]];
        torch::Tensor joint_top = skeleton[0][0][zaxis[1]];
        torch::Tensor temp = torch::eye(3)[2];
        torch::Tensor axis = torch::cross(joint_bottom - joint_top, temp);
        torch::Tensor angle = angle_between(joint_top - joint_bottom, temp);
        torch::Tensor matrix_z = rotation_matrix(axis, angle);
        for (int i_p = 0; i_p < skeleton.sizes()[0]; i_p++){
            torch::Tensor person = skeleton[i_p];
            if (!person.sum().item().toBool())
                continue;
            for (int i_f = 0; i_f < person.sizes()[0]; i_f++){
                torch::Tensor frame = person[i_f];
                if (!frame.sum().item().toBool())
                    continue;
                for (int i_j = 0; i_j < frame.sizes()[0]; i_j++){
                    torch::Tensor joint = frame[i_j];
                    s[i_s][i_p][i_f][i_j] = torch::mm(matrix_z, joint.reshape({joint.sizes()[0], 1})).reshape({joint.sizes()[0]});
                }
            }
        }
    }

    for (int i_s = 0; i_s < s.sizes()[0]; i_s++){
        torch::Tensor skeleton = s[i_s];
        if (!skeleton.sum().item().toBool())
            continue;
        torch::Tensor joint_rshoulder = skeleton[0][0][xaxis[0]];
        torch::Tensor joint_lshoulder = skeleton[0][0][xaxis[1]];
        torch::Tensor temp = torch::eye(3)[0];
        torch::Tensor axis = torch::cross(joint_rshoulder - joint_lshoulder, temp);
        torch::Tensor angle = angle_between(joint_rshoulder - joint_lshoulder, temp);
        torch::Tensor matrix_x = rotation_matrix(axis, angle);
        for (int i_p = 0; i_p < skeleton.sizes()[0]; i_p++){
            torch::Tensor person = skeleton[i_p];
            if (!person.sum().item().toBool())
                continue;
            for (int i_f = 0; i_f < person.sizes()[0]; i_f++){
                torch::Tensor frame = person[i_f];
                if (!frame.sum().item().toBool())
                    continue;
                for (int i_j = 0; i_j < frame.sizes()[0]; i_j++){
                    torch::Tensor joint = frame[i_j];
                    s[i_s][i_p][i_f][i_j] = torch::mm(matrix_x, joint.reshape({joint.sizes()[0], 1})).reshape({joint.sizes()[0]});
                }
            }
        }
    }

    data = s.permute({0, 4, 2, 3, 1});

    return data;
}

vector<int> ActRecognizer::linearize(const vector<vector<vector<int>>> &vec_vec) {
    vector<int> vec;
    for (const auto& u : vec_vec) {
        for (const auto& v : u) {
            for (auto &w : v) {
                vec.push_back(w);
            }
        }
    }
    return vec;
}
