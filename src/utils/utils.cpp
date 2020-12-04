//
// Created by ljh on 2020/10/12.
//
#include "utils/utils.h"
using namespace std;
namespace ljh {
    unsigned char *RGB2BGR(const string &picPath, int resize_w, int resize_h) {
        cv::Mat img = cv::imread(picPath);
        if (img.data == nullptr) {
            cout << "Images not Found" << endl;
            return nullptr;
        }
        cv::resize(img, img, cv::Size(resize_w, resize_h));
        auto *data = (unsigned char *) img.data;
        int step = img.step;
        int h = resize_h;
        int w = resize_w;
        int c = img.channels();

        auto *bgr = (unsigned char *) malloc(w * h * c);

        for (int k = 0; k < c; k++)
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                    bgr[w * h * k + w * i + j] = data[i * step + j * c + k];

        return bgr;
    }

    void saveBGR(const string &savPath, const unsigned char *bgr, int width, int height, int channels, int dst_w,
                 int dst_h) {
        if (bgr == nullptr) {
            cout << "bgr is empty" << endl;
            return;
        }
        unsigned char conv_data[width * height * channels];
        for (int h = 0; h < height; h++)
            for (int w = 0; w < width; w++)
                for (int c = 0; c < channels; c++)
                    conv_data[h * width * channels + w * channels + c] = bgr[(height * width) * c + h * width + w];

        cv::Mat rgb = cv::Mat(height, width, CV_8UC3, conv_data);
        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(dst_w, dst_h));
        cv::imwrite(savPath, resized);
    }

    void GetFileNames(const string &path, vector<string> &filenames_) {
        DIR *pDir;
        struct dirent *ptr;
        if (!(pDir = opendir(path.c_str()))) {
            cout << "Folder doesn't Exist!" << endl;
            return;
        }
        while ((ptr = readdir(pDir)) != nullptr) {
            if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
                filenames_.emplace_back(ptr->d_name);
        }
        closedir(pDir);
    }
}