#ifndef _FACE_DETECT_H_
#define _FACE_DETECT_H_

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using std::string;
using std::vector;

namespace TIEVD{


typedef struct FaceBox {
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	float score;
} FaceBox;
typedef struct FaceInfo {
	float bbox_reg[4];
	float landmark_reg[10];
	float landmark[10];
	FaceBox bbox;
} FaceInfo;

class FaceDetect {
public:
	FaceDetect(const string& proto_model_dir, float threhold_p=0.7f, float threhold_r=0.8f, float threhold_o = 0.8f, float factor = 0.709f);
	std::vector<FaceInfo> Detect(const cv::Mat& img,  const int min_face = 64 , const int stage = 3);
	std::vector<FaceInfo> Detect_MaxFace(const cv::Mat& img,  const int min_face= 64,  const int stage = 3);
	~FaceDetect();
private:
    int threads_num = 2;

};

}









#endif // _FaceDetect_H_

