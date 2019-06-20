#include <opencv2/opencv.hpp>
#include<string>
#include <thread>
#include "face_detect.h"
using namespace cv;

void face_detect1(){

    cv::Mat img1 = cv::imread("./img/001.jpg");

    //检测 人脸的对象。
    std::string model_path = "./models/";
    TIEVD::FaceDetect face_detect(model_path, 0.7f, 0.8f, 0.9f);

    std::vector<TIEVD::FaceInfo> face_info1 = face_detect.Detect_MaxFace(img1, 32, 3);
    std::vector<TIEVD::FaceInfo> face_info2 = face_detect.Detect(img1, 32, 3);

    std::cout<<"face_info1:"<<face_info1.size()<<":"<<face_info2.size()<<std::endl;

}



int main(){

	face_detect1();


	return 0;
}
