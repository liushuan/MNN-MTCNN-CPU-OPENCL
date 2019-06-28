
#include "face_detect.h"

#include <Interpreter.hpp>
#include <MNNDefine.h>
#include <Tensor.hpp>
#include <ImageProcess.hpp>

namespace TIEVD{

using namespace MNN;
using namespace MNN::CV;

std::shared_ptr<MNN::Interpreter> PNet_ = NULL;
std::shared_ptr<MNN::Interpreter> RNet_ = NULL;
std::shared_ptr<MNN::Interpreter> ONet_ = NULL;

MNN::Session * sess_p = NULL;
MNN::Session * sess_r = NULL;
MNN::Session * sess_o = NULL;

MNN::Tensor * p_input = nullptr;
MNN::Tensor * p_out_pro = nullptr;
MNN::Tensor * p_out_reg = nullptr;

MNN::Tensor * r_input = nullptr;
MNN::Tensor * r_out_pro = nullptr;
MNN::Tensor * r_out_reg = nullptr;

MNN::Tensor * o_input = nullptr;
MNN::Tensor * o_out_pro = nullptr;
MNN::Tensor * o_out_reg = nullptr;
MNN::Tensor * o_out_lank = nullptr;

std::shared_ptr<ImageProcess> pretreat_data;

std::vector<FaceInfo> candidate_boxes_;
std::vector<FaceInfo> total_boxes_;

static float threhold_p = 0.8f;
static float threhold_r = 0.8f;
static float threhold_o = 0.9f;
static float iou_threhold = 0.7f;
static float factor = 0.709f;
//static int min_face = 48;

//pnet config
static const float pnet_stride = 2;
static const float pnet_cell_size = 12;
static const int pnet_max_detect_num = 5000;
//mean & std
static const float mean_val = 127.5f;
static const float std_val = 0.0078125f;


static bool CompareBBox(const FaceInfo & a, const FaceInfo & b) {
	return a.bbox.score > b.bbox.score;
}


static float IoU(float xmin, float ymin, float xmax, float ymax,
	float xmin_, float ymin_, float xmax_, float ymax_, bool is_iom) {
	float iw = std::min(xmax, xmax_) - std::max(xmin, xmin_) + 1;
	float ih = std::min(ymax, ymax_) - std::max(ymin, ymin_) + 1;
	if (iw <= 0 || ih <= 0)
		return 0;
	float s = iw*ih;
	if (is_iom) {
		float ov = s / std::min((xmax - xmin + 1)*(ymax - ymin + 1), (xmax_ - xmin_ + 1)*(ymax_ - ymin_ + 1));
		return ov;
	}
	else {
		float ov = s / ((xmax - xmin + 1)*(ymax - ymin + 1) + (xmax_ - xmin_ + 1)*(ymax_ - ymin_ + 1) - s);
		return ov;
	}
}

 static std::vector<FaceInfo> NMS(std::vector<FaceInfo>& bboxes,
	float thresh, char methodType) {
	std::vector<FaceInfo> bboxes_nms;
	if (bboxes.size() == 0) {
		return bboxes_nms;
	}
	std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

	int32_t select_idx = 0;
	int32_t num_bbox = static_cast<int32_t>(bboxes.size());
	std::vector<int32_t> mask_merged(num_bbox, 0);
	bool all_merged = false;

	while (!all_merged) {
		while (select_idx < num_bbox && mask_merged[select_idx] == 1)
			select_idx++;
		if (select_idx == num_bbox) {
			all_merged = true;
			continue;
		}
		bboxes_nms.push_back(bboxes[select_idx]);
		mask_merged[select_idx] = 1;

		FaceBox select_bbox = bboxes[select_idx].bbox;
		float area1 = static_cast<float>((select_bbox.xmax - select_bbox.xmin + 1) * (select_bbox.ymax - select_bbox.ymin + 1));
		float x1 = static_cast<float>(select_bbox.xmin);
		float y1 = static_cast<float>(select_bbox.ymin);
		float x2 = static_cast<float>(select_bbox.xmax);
		float y2 = static_cast<float>(select_bbox.ymax);

		select_idx++;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_num)
#endif
		for (int32_t i = select_idx; i < num_bbox; i++) {
			if (mask_merged[i] == 1)
				continue;

			FaceBox & bbox_i = bboxes[i].bbox;
			float x = std::max<float>(x1, static_cast<float>(bbox_i.xmin));
			float y = std::max<float>(y1, static_cast<float>(bbox_i.ymin));
			float w = std::min<float>(x2, static_cast<float>(bbox_i.xmax)) - x + 1;
			float h = std::min<float>(y2, static_cast<float>(bbox_i.ymax)) - y + 1;
			if (w <= 0 || h <= 0)
				continue;

			float area2 = static_cast<float>((bbox_i.xmax - bbox_i.xmin + 1) * (bbox_i.ymax - bbox_i.ymin + 1));
			float area_intersect = w * h;

			switch (methodType) {
			case 'u':
				if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > thresh)
					mask_merged[i] = 1;
				break;
			case 'm':
				if (static_cast<float>(area_intersect) / std::min(area1, area2) > thresh)
					mask_merged[i] = 1;
				break;
			default:
				break;
			}
		}
	}
	return bboxes_nms;
}
 static void BBoxRegression(vector<FaceInfo>& bboxes) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_num)
#endif
	for (int i = 0; i < bboxes.size(); ++i) {
		FaceBox &bbox = bboxes[i].bbox;
		float *bbox_reg = bboxes[i].bbox_reg;
		float w = bbox.xmax - bbox.xmin + 1;
		float h = bbox.ymax - bbox.ymin + 1;
		bbox.xmin += bbox_reg[0] * w;
		bbox.ymin += bbox_reg[1] * h;
		bbox.xmax += bbox_reg[2] * w;
		bbox.ymax += bbox_reg[3] * h;
	}
}
 static void BBoxPad(vector<FaceInfo>& bboxes, int width, int height) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_num)
#endif
	for (int i = 0; i < bboxes.size(); ++i) {
		FaceBox &bbox = bboxes[i].bbox;
		bbox.xmin = round(std::max(bbox.xmin, 0.f));
		bbox.ymin = round(std::max(bbox.ymin, 0.f));
		bbox.xmax = round(std::min(bbox.xmax, width - 1.f));
		bbox.ymax = round(std::min(bbox.ymax, height - 1.f));
	}
}
 static void BBoxPadSquare(vector<FaceInfo>& bboxes, int width, int height) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_num)
#endif
	for (int i = 0; i < bboxes.size(); ++i) {
		FaceBox &bbox = bboxes[i].bbox;
		float w = bbox.xmax - bbox.xmin + 1;
		float h = bbox.ymax - bbox.ymin + 1;
		float side = h>w ? h : w;
		bbox.xmin = round(std::max(bbox.xmin + (w - side)*0.5f, 0.f));
		bbox.ymin = round(std::max(bbox.ymin + (h - side)*0.5f, 0.f));
		bbox.xmax = round(std::min(bbox.xmin + side - 1, width - 1.f));
		bbox.ymax = round(std::min(bbox.ymin + side - 1, height - 1.f));
	}
}
 static void GenerateBBox(float * confidence_data, float *reg_box, int feature_map_w_, int feature_map_h_, float scale, float thresh) {

	int spatical_size = feature_map_w_*feature_map_h_;

	candidate_boxes_.clear();
	float v_scale = 1.0/scale;
	for (int i = 0; i<spatical_size; ++i) {
        int stride = i<<2;
		if (confidence_data[stride + 1] >= thresh) {
			int y = i / feature_map_w_;
			int x = i - feature_map_w_ * y;
			FaceInfo faceInfo;
			FaceBox &faceBox = faceInfo.bbox;

			faceBox.xmin = (float)(x * pnet_stride) * v_scale;
			faceBox.ymin = (float)(y * pnet_stride) * v_scale;
			faceBox.xmax = (float)(x * pnet_stride + pnet_cell_size - 1.f) * v_scale;
			faceBox.ymax = (float)(y * pnet_stride + pnet_cell_size - 1.f) * v_scale;

			faceInfo.bbox_reg[0] = reg_box[stride];
			faceInfo.bbox_reg[1] = reg_box[stride + 1];
			faceInfo.bbox_reg[2] = reg_box[stride + 2];
			faceInfo.bbox_reg[3] = reg_box[stride + 3];

			faceBox.score = confidence_data[stride + 1];
			candidate_boxes_.push_back(faceInfo);
		}
	}
}

FaceDetect::FaceDetect(const string& proto_model_dir, float threhold_p_, float threhold_r_, float threhold_o_, float factor_){
	threhold_p = threhold_p_;
	threhold_r = threhold_r_;
	threhold_o = threhold_o_;
	factor = factor_;
    threads_num = 2;
	PNet_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile((proto_model_dir + "det1.mnn").c_str()));

	RNet_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile((proto_model_dir + "det2.mnn").c_str()));

	ONet_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile((proto_model_dir + "det3-half.mnn").c_str()));


    MNN::ScheduleConfig config;
    config.type = (MNNForwardType)0;
    config.numThread = 1; // 1 faster

    BackendConfig backendConfig;
    backendConfig.precision = BackendConfig::Precision_Low;
    backendConfig.power = BackendConfig::Power_High;
    config.backendConfig = &backendConfig;

    sess_p =  PNet_->createSession(config);
    sess_r =  RNet_->createSession(config);
    sess_o =  ONet_->createSession(config);


    p_input = PNet_->getSessionInput(sess_p, NULL);
    p_out_pro = PNet_->getSessionOutput(sess_p, "prob1");
    p_out_reg = PNet_->getSessionOutput(sess_p, "conv4-2");

    r_input = RNet_->getSessionInput(sess_r, NULL);
    r_out_pro = RNet_->getSessionOutput(sess_r, "prob1");
    r_out_reg = RNet_->getSessionOutput(sess_r, "conv5-2");

    o_input = ONet_->getSessionInput(sess_o, NULL);
    o_out_pro = ONet_->getSessionOutput(sess_o, "prob1");
    o_out_reg = ONet_->getSessionOutput(sess_o, "conv6-2");
    o_out_lank = ONet_->getSessionOutput(sess_o, "conv6-3");

    ImageProcess::Config config_data;
    config_data.filterType = BILINEAR;
    const float mean_vals[3] = {mean_val, mean_val, mean_val};
    const float norm_vals[3] = {std_val, std_val, std_val};
    ::memcpy(config_data.mean, mean_vals, sizeof(mean_vals));
    ::memcpy(config_data.normal, norm_vals, sizeof(norm_vals));
    config_data.sourceFormat = RGBA;
    config_data.destFormat = BGR;

    pretreat_data = std::shared_ptr<ImageProcess>(ImageProcess::create(config_data));


}

FaceDetect::~FaceDetect() {
	PNet_->releaseModel();
	RNet_->releaseModel();
	ONet_->releaseModel();
	candidate_boxes_.clear();
	total_boxes_.clear();
}

uint8_t* get_img(cv::Mat img){
    uchar * colorData = new uchar[img.total() * 4];
    cv::Mat MatTemp(img.size(), CV_8UC4, colorData);
    cv::cvtColor(img, MatTemp, CV_BGR2RGBA, 4);
    return (uint8_t *)MatTemp.data;
}

static vector<FaceInfo> ProposalNet(const cv::Mat& img, int minSize, float threshold, float factor) {
	int width = img.cols;
	int height = img.rows;
	float scale = 12.0f / minSize;
	float minWH = std::min(height, width) *scale;
	std::vector<float> scales;
	while (minWH >= 12) {
		scales.push_back(scale);
		minWH *= factor;
		scale *= factor;
	}
	total_boxes_.clear();

	uint8_t *pImg = get_img(img);
	for (int i = 0; i < scales.size(); i++) {
		int ws = (int)std::ceil(width*scales[i]);
		int hs = (int)std::ceil(height*scales[i]);
        std::vector<int> inputDims = {1, 3, hs, ws};
        PNet_->resizeTensor(p_input, inputDims);
        PNet_->resizeSession(sess_p);

		MNN::CV::Matrix trans;
		trans.postScale(1.0f/ws, 1.0f/hs);
		trans.postScale(width, height);
        pretreat_data->setMatrix(trans);
        pretreat_data->convert(pImg, width, height, 0, p_input);

		PNet_->runSession(sess_p);
        float * confidence = p_out_pro->host<float>();
        float * reg = p_out_reg->host<float>();

        int feature_w = p_out_pro->width();
        int feature_h = p_out_pro->height();

		GenerateBBox(confidence, reg, feature_w, feature_h, scales[i], threshold);
		std::vector<FaceInfo> bboxes_nms = NMS(candidate_boxes_, 0.5f, 'u');
		if (bboxes_nms.size() > 0) {
			total_boxes_.insert(total_boxes_.end(), bboxes_nms.begin(), bboxes_nms.end());
		}
	}

	int num_box = (int)total_boxes_.size();
	vector<FaceInfo> res_boxes;
	if (num_box != 0) {
		res_boxes = NMS(total_boxes_, 0.5f, 'u');
		BBoxRegression(res_boxes);
		BBoxPadSquare(res_boxes, width, height);
	}
        delete pImg;
	return res_boxes;
}

static std::vector<FaceInfo> NextStage(const cv::Mat& image, vector<FaceInfo> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold) {
    vector<FaceInfo> res;
	int batch_size = pre_stage_res.size();

	switch (stage_num) {
	case 2: {

		for (int n = 0; n < batch_size; ++n)
		{
			FaceBox &box = pre_stage_res[n].bbox;
			cv::Mat roi = image(cv::Rect(cv::Point((int)box.xmin, (int)box.ymin), cv::Point((int)box.xmax, (int)box.ymax))).clone();

            //cv::imshow("face", roi);
            //cv::waitKey(0);

			MNN::CV::Matrix trans;
            trans.postScale(1.0/input_w, 1.0/input_h);
            trans.postScale(roi.cols, roi.rows);
            pretreat_data->setMatrix(trans);

            uint8_t *pImg = get_img(roi);
            pretreat_data->convert(pImg, roi.cols, roi.rows, 0, r_input);
            delete pImg;
		    RNet_->runSession(sess_r);

            float * confidence = r_out_pro->host<float>();
            float * reg_box = r_out_reg->host<float>();

			float conf = confidence[1];
			if (conf >= threshold) {
				FaceInfo info;
				info.bbox.score = conf;
				info.bbox.xmin = pre_stage_res[n].bbox.xmin;
				info.bbox.ymin = pre_stage_res[n].bbox.ymin;
				info.bbox.xmax = pre_stage_res[n].bbox.xmax;
				info.bbox.ymax = pre_stage_res[n].bbox.ymax;
				for (int i = 0; i < 4; ++i) {
					info.bbox_reg[i] = reg_box[i];
				}
                res.push_back(info);

			}
		}
		break;
	}
	case 3:{
//#ifdef _OPENMP
//#pragma omp parallel for num_threads(threads_num)
//#endif
		for (int n = 0; n < batch_size; ++n)
		{
			FaceBox &box = pre_stage_res[n].bbox;
			cv::Mat roi = image(cv::Rect(cv::Point((int)box.xmin, (int)box.ymin), cv::Point((int)box.xmax, (int)box.ymax))).clone();

            //cv::imshow("face", roi);
            //cv::waitKey(0);

            MNN::CV::Matrix trans;
            trans.postScale(1.0f/input_w, 1.0f/input_h);
            trans.postScale(roi.cols, roi.rows);
            pretreat_data->setMatrix(trans);
            uint8_t *pImg = get_img(roi);
            pretreat_data->convert(pImg, roi.cols, roi.rows, 0, o_input);
            delete pImg;
		    ONet_->runSession(sess_o);
            float * confidence = o_out_pro->host<float>();
            float * reg_box = o_out_reg->host<float>();
            float * reg_landmark = o_out_lank->host<float>();

			float conf = confidence[1];
            //std::cout<<"stage three:"<<confidence[0]<<" "<<confidence[1]<<" "<<confidence[2]<<" "<<confidence[4]<<std::endl;
			if (conf >= threshold) {
				FaceInfo info;
				info.bbox.score = conf;
				info.bbox.xmin = pre_stage_res[n].bbox.xmin;
				info.bbox.ymin = pre_stage_res[n].bbox.ymin;
				info.bbox.xmax = pre_stage_res[n].bbox.xmax;
				info.bbox.ymax = pre_stage_res[n].bbox.ymax;
				for (int i = 0; i < 4; ++i) {
					info.bbox_reg[i] = reg_box[i];
				}
				float w = info.bbox.xmax - info.bbox.xmin + 1.f;
				float h = info.bbox.ymax - info.bbox.ymin + 1.f;
				for (int i = 0; i < 5; ++i) {
					info.landmark[2 * i] = reg_landmark[2 * i] * w + info.bbox.xmin;
					info.landmark[2 * i + 1] = reg_landmark[2 * i + 1] * h + info.bbox.ymin;
				}
                res.push_back(info);
			}
		}
		break;
	}
	default:
		return res;
		break;
	}
	return res;
}

vector<FaceInfo> FaceDetect::Detect(const cv::Mat& image,  const int min_face,  const int stage) {

	vector<FaceInfo> pnet_res;
	vector<FaceInfo> rnet_res;
	vector<FaceInfo> onet_res;

	if (stage >= 1) {
		pnet_res = ProposalNet(image, min_face, threhold_p, factor);
	}
    //std::cout<<"p size is:"<<pnet_res.size()<<std::endl;
	if (stage >= 2 && pnet_res.size()>0) {
		if (pnet_max_detect_num < (int)pnet_res.size()) {
			pnet_res.resize(pnet_max_detect_num);
		}
		rnet_res = NextStage(image, pnet_res, 24, 24, 2, threhold_r);
		rnet_res = NMS(rnet_res, iou_threhold, 'u');
		BBoxRegression(rnet_res);
		BBoxPadSquare(rnet_res, image.cols, image.rows);
	}
    //std::cout<<"r size is:"<<rnet_res.size()<<std::endl;
	if (stage >= 3 && rnet_res.size()>0) {
		onet_res = NextStage(image, rnet_res, 48, 48, 3, threhold_o);
		BBoxRegression(onet_res);
		onet_res = NMS(onet_res, iou_threhold, 'm');
		BBoxPad(onet_res, image.cols, image.rows);
	}
	if (stage == 1) {
		return pnet_res;
	}
	else if (stage == 2) {
		return rnet_res;
	}
	else if (stage == 3) {
		return onet_res;
	}
	else {
		return onet_res;
	}
}


static std::vector<FaceInfo> extractMaxFace(std::vector<FaceInfo> boundingBox_)
{
	if (boundingBox_.empty()) {
		return std::vector<FaceInfo>{};
	}
/*
	sort(boundingBox_.begin(), boundingBox_.end(), CompareBBox);
	for (std::vector<FaceInfo>::iterator itx = boundingBox_.begin() + 1; itx != boundingBox_.end();) {
		itx = boundingBox_.erase(itx);
	}
*/
	float max_area = 0;
	int index = 0;
	for (int i = 0; i < boundingBox_.size(); ++i){
 		FaceBox select_bbox = boundingBox_[i].bbox;
                float area1 = static_cast<float>((select_bbox.xmax - select_bbox.xmin + 1) * (select_bbox.ymax - select_bbox.ymin + 1));
		if (area1 > max_area){
			max_area = area1;
			index = i;
		}
	}
	return std::vector<FaceInfo>{boundingBox_[index]};	
}

std::vector<FaceInfo> FaceDetect::Detect_MaxFace(const cv::Mat& img, const int min_face,  const int stage){
    vector<FaceInfo> pnet_res;
	vector<FaceInfo> rnet_res;
	vector<FaceInfo> onet_res;

    //total_boxes_.clear();
    //candidate_boxes_.clear();

	int width = img.cols;
	int height = img.rows;
	float scale = 12.0f / min_face;
	float minWH = std::min(height, width) *scale;
	std::vector<float> scales;
	while (minWH >= 12) {
		scales.push_back(scale);
		minWH *= factor;
		scale *= factor;
	}

    //sort(scales.begin(), scales.end());
    std::reverse(scales.begin(), scales.end());

	uint8_t *pImg = get_img(img);
	for (int i = 0; i < scales.size(); i++) {
		int ws = (int)std::ceil(width*scales[i]);
		int hs = (int)std::ceil(height*scales[i]);
        std::vector<int> inputDims = {1, 3, hs, ws};
        PNet_->resizeTensor(p_input, inputDims);
        PNet_->resizeSession(sess_p);

		MNN::CV::Matrix trans;
		trans.postScale(1.0f/ws, 1.0f/hs);
		trans.postScale(width, height);
        pretreat_data->setMatrix(trans);
        pretreat_data->convert(pImg, width, height, 0, p_input);

		PNet_->runSession(sess_p);
        float * confidence = p_out_pro->host<float>();
        float * reg = p_out_reg->host<float>();

        int feature_w = p_out_pro->width();
        int feature_h = p_out_pro->height();

		GenerateBBox(confidence, reg, feature_w, feature_h, scales[i], threhold_p);
		std::vector<FaceInfo> bboxes_nms = NMS(candidate_boxes_, 0.5f, 'u');

		//nmsTwoBoxs(bboxes_nms, pnet_res, 0.5);
		if (bboxes_nms.size() > 0) {
			pnet_res.insert(pnet_res.end(), bboxes_nms.begin(), bboxes_nms.end());
		}else{
            continue;
		}
		BBoxRegression(pnet_res);
		BBoxPadSquare(pnet_res, width, height);

        bboxes_nms.clear();
        bboxes_nms = NextStage(img, pnet_res, 24, 24, 2, threhold_r);
		bboxes_nms = NMS(bboxes_nms, iou_threhold, 'u');
		//nmsTwoBoxs(bboxes_nms, rnet_res, 0.5)
		if (bboxes_nms.size() > 0) {
			rnet_res.insert(rnet_res.end(), bboxes_nms.begin(), bboxes_nms.end());
		}else{
            pnet_res.clear();
            continue;
		}
		BBoxRegression(rnet_res);
		BBoxPadSquare(rnet_res, img.cols, img.rows);


        onet_res = NextStage(img, rnet_res, 48, 48, 3, threhold_r);

        BBoxRegression(onet_res);
		onet_res = NMS(onet_res, iou_threhold, 'm');
		BBoxPad(onet_res, img.cols, img.rows);

        if(onet_res.size() < 1){
            pnet_res.clear();
            rnet_res.clear();
            continue;
        }else{
            onet_res =  extractMaxFace(onet_res);
            delete pImg;
            return onet_res;
        }
	}
	delete pImg;
	return std::vector<FaceInfo>{};
}

}
