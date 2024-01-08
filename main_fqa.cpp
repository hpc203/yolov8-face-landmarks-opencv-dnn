#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

typedef struct
{
	cv::Rect rect;
	float prob;
	vector<Point> kpt;
} face;

class YOLOv8_face
{
public:
	YOLOv8_face(string modelpath, float confThreshold, float nmsThreshold);
	vector<face> detect(Mat& frame);
private:
	Mat resize_image(Mat srcimg, int *newh, int *neww, int *padh, int *padw);
	const bool keep_ratio = true;
	const int inpWidth = 640;
	const int inpHeight = 640;
	float confThreshold;
	float nmsThreshold;
	const int num_class = 1;  ///只有人脸这一个类别
	const int reg_max = 16;
	Net net;
	void softmax_(const float* x, float* y, int length);
	void generate_proposal(Mat out, vector<Rect>& boxes, vector<float>& confidences, vector< vector<Point>>& landmarks, int imgh, int imgw, float ratioh, float ratiow, int padh, int padw);
};

static inline float sigmoid_x(float x)
{
	return static_cast<float>(1.f / (1.f + exp(-x)));
}

YOLOv8_face::YOLOv8_face(string modelpath, float confThreshold, float nmsThreshold)
{
	this->confThreshold = confThreshold;
	this->nmsThreshold = nmsThreshold;
	this->net = readNet(modelpath);
}

Mat YOLOv8_face::resize_image(Mat srcimg, int *newh, int *neww, int *padh, int *padw)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*padw = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *padw, this->inpWidth - *neww - *padw, BORDER_CONSTANT, 0);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*padh = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *padh, this->inpHeight - *newh - *padh, 0, 0, BORDER_CONSTANT, 0);
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void YOLOv8_face::softmax_(const float* x, float* y, int length)
{
	float sum = 0;
	int i = 0;
	for (i = 0; i < length; i++)
	{
		y[i] = exp(x[i]);
		sum += y[i];
	}
	for (i = 0; i < length; i++)
	{
		y[i] /= sum;
	}
}

void YOLOv8_face::generate_proposal(Mat out, vector<Rect>& boxes, vector<float>& confidences, vector< vector<Point>>& landmarks, int imgh,int imgw, float ratioh, float ratiow, int padh, int padw)
{
	const int feat_h = out.size[2];
	const int feat_w = out.size[3];
	cout << out.size[1] << "," << out.size[2] << "," << out.size[3] << endl;
	const int stride = (int)ceil((float)inpHeight / feat_h);
	const int area = feat_h * feat_w;
	float* ptr = (float*)out.data;
	float* ptr_cls = ptr + area * reg_max * 4;
	float* ptr_kp = ptr + area * (reg_max * 4 + num_class);

	for (int i = 0; i < feat_h; i++)
	{
		for (int j = 0; j < feat_w; j++)
		{
			const int index = i * feat_w + j;
			int cls_id = -1;
			float max_conf = -10000;
			for (int k = 0; k < num_class; k++)
			{
				float conf = ptr_cls[k*area + index];
				if (conf > max_conf)
				{
					max_conf = conf;
					cls_id = k;
				}
			}
			float box_prob = sigmoid_x(max_conf);
			if (box_prob > this->confThreshold)
			{
				float pred_ltrb[4];
				float* dfl_value = new float[reg_max];
				float* dfl_softmax = new float[reg_max];
				for (int k = 0; k < 4; k++)
				{
					for (int n = 0; n < reg_max; n++)
					{
						dfl_value[n] = ptr[(k*reg_max + n)*area + index];
					}
					softmax_(dfl_value, dfl_softmax, reg_max);

					float dis = 0.f;
					for (int n = 0; n < reg_max; n++)
					{
						dis += n * dfl_softmax[n];
					}

					pred_ltrb[k] = dis * stride;
				}
				float cx = (j + 0.5f)*stride;
				float cy = (i + 0.5f)*stride;
				float xmin = max((cx - pred_ltrb[0] - padw)*ratiow, 0.f);  ///还原回到原图
				float ymin = max((cy - pred_ltrb[1] - padh)*ratioh, 0.f);
				float xmax = min((cx + pred_ltrb[2] - padw)*ratiow, float(imgw - 1));
				float ymax = min((cy + pred_ltrb[3] - padh)*ratioh, float(imgh - 1));
				Rect box = Rect(int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin));
				boxes.push_back(box);
				confidences.push_back(box_prob);

				vector<Point> kpts(5);
				for (int k = 0; k < 5; k++)
				{
					float x = ((ptr_kp[(k * 3)*area + index] * 2 + j)*stride - padw)*ratiow;  ///还原回到原图
					float y = ((ptr_kp[(k * 3 + 1)*area + index] * 2 + i)*stride - padh)*ratioh;
					///float pt_conf = sigmoid_x(ptr_kp[(k * 3 + 2)*area + index]);
					kpts[k] = Point(int(x), int(y));
				}
				landmarks.push_back(kpts);
			}
		}
	}
}


vector<face> YOLOv8_face::detect(Mat& srcimg)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat dst = this->resize_image(srcimg, &newh, &neww, &padh, &padw);
	Mat blob;
	blobFromImage(dst, blob, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
	this->net.setInput(blob);
	vector<Mat> outs;
	net.enableWinograd(false);  ////如果是opencv4.7，那就需要加上这一行
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

	/////generate proposals
	vector<Rect> boxes;
	vector<float> confidences;
	vector< vector<Point>> landmarks;
	float ratioh = (float)srcimg.rows / newh, ratiow = (float)srcimg.cols / neww;

	generate_proposal(outs[0], boxes, confidences, landmarks, srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);
	generate_proposal(outs[1], boxes, confidences, landmarks, srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);
	generate_proposal(outs[2], boxes, confidences, landmarks, srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	vector<face> face_boxes;
	for (size_t i = 0; i < indices.size(); ++i)
	{
		const int idx = indices[i];
		face_boxes.push_back({ boxes[idx], confidences[idx], landmarks[idx] });
	}
	return face_boxes;
}

class face_quality_assessment
{
public:
	face_quality_assessment(string modelpath)
	{
		this->net = readNet(modelpath);
	}
	float detect(Mat cropped);
private:
	const int inpWidth = 112;
	const int inpHeight = 112;
	Net net;
	Mat normalize_(Mat img);
	const float mean_[3] = { 0.5, 0.5, 0.5 };
	const float std_[3] = { 0.5, 0.5, 0.5 };
};

Mat face_quality_assessment::normalize_(Mat img)
{
	vector<cv::Mat> bgrChannels(3);
	split(img, bgrChannels);
	for (int c = 0; c < 3; c++)
	{
		bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1.0 / (255.0* std_[c]), (0.0 - mean_[c]) / std_[c]);
	}
	Mat m_normalized_mat;
	merge(bgrChannels, m_normalized_mat);
	return m_normalized_mat;
}

float face_quality_assessment::detect(Mat cropped)
{
	Mat rgbimg;
	cvtColor(cropped, rgbimg, COLOR_BGR2RGB);
	resize(rgbimg, rgbimg, cv::Size(this->inpWidth, this->inpHeight));
	Mat normalized_mat = this->normalize_(rgbimg);
	Mat blob = blobFromImage(normalized_mat);

	this->net.setInput(blob);
	vector<Mat> outs;
	net.enableWinograd(false);  ////如果是opencv4.7，那就需要加上这一行
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
	float* pdata = (float*)outs[0].data;  ///形状1x10
	const int length = outs[0].size[1];
	float fqa_prob_mean = 0;
	for (int i = 0; i < length; i++)
	{
		fqa_prob_mean += pdata[i];
	}
	fqa_prob_mean /= length;
	return fqa_prob_mean;
}

int main()
{
	YOLOv8_face face_detector("weights/yolov8n-face.onnx", 0.45, 0.5);
	face_quality_assessment fqa("weights/face-quality-assessment.onnx");

	string imgpath = "images/1.jpg";
	Mat srcimg = imread(imgpath);
	vector<face> face_boxes = face_detector.detect(srcimg);

	Mat drawimg = srcimg.clone();
	for (int i = 0; i < face_boxes.size(); i++)
	{
		Mat crop_img = srcimg(face_boxes[i].rect);
		const float fqa_prob_mean = fqa.detect(crop_img);
		cv::rectangle(drawimg, face_boxes[i].rect, cv::Scalar(0, 0, 255), 2);
		string label = format("fqa_score:%.2f", fqa_prob_mean);
		cv::putText(drawimg, label, cv::Point(face_boxes[i].rect.x, face_boxes[i].rect.y-5), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
	}
	
	static const string kWinName = "Deep learning face-quality-assessment use OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, drawimg);
	waitKey(0);
	destroyAllWindows();
}
