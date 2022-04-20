#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include <stdio.h>
#include "direct.h"

//using namespace std;
cv::Mat frame, frame1, gray, bin, inputTarget, res, res1, inputTarget_bin, inputTarget_bininv;
cv::Mat roi_start, roi_end, roi_diff;
cv::Mat roi_diffbin;
//cv::VideoCapture capture, capture1;
double minVal, maxVal;
cv::Point Loc;
bool ones;
#include <iostream>
#include <iomanip> 
#include "Poco/Net/StreamSocket.h"
#include "Poco/Net/SocketAddress.h"
#include <string>
#include <vector>
#include <fstream>
#include <thread>
#include <atomic>
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable
#include <sys/types.h>
#include <time.h>
#include <WinBase.h>
#include <windows.h>
#include <iostream>
#include "dirent.h"
#include <atlstr.h>
#include "Poco\Data\ODBC\Connector.h"
#include "Poco\Data\Session.h"
#include "Poco\Util\AbstractConfiguration.h"
#include "Poco\Util\IniFileConfiguration.h"
#define OPENCV
#include "yolo_v2_class.hpp"	// imported functions from DLL
#include<filesystem>


using Poco::Net::SocketAddress;
using Poco::Net::StreamSocket;
using Poco::AutoPtr;
using Poco::Util::IniFileConfiguration;
using namespace Poco::Data::Keywords;
using Poco::Data::Session;
using Poco::Data::Statement;

std::string inputFilename;
std::string saveFilename;
std::string testsaveImage;
std::string image_Name;
std::string imagePath;
std::string picfilename_single = "";
std::string picfilename_double1 = "";
std::string picfilename_double2 = "";
std::string imagePath_single;
std::string imagePath_double1;
std::string imagePath_double2;
std::string Xray_result, deepColorResult;

//config.ini
///////////////////////
//part of XrayAicheck//
///////////////////////
std::string dbString;
double thresholdProb = 0.0;
double thresholdProb2 = 0.0;
double thresholdProbClass[14] = {0.0, 0.0, 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 };
int largest_area_thr;
int debugMode;
std::string test_Path;
////////////////////////
//part of FrameCapture//
////////////////////////
std::string capture0_state;
std::string capture1_state;
std::string LtoR;
std::string folder_path = "";
int duration = 60;
int xraytype = 0;
int bottom_limit;
int thr_line;
int showpic;
int video_col;
int video_row;
double maxVal_thr; //cv::Match Template_threshold
int min_dist;//object min col
double capture_min_time;
int frame_height, frame_width; //frame長寬
int frame_coordinate_x, frame_coordinate_y; //frame起始座標
int inputTarget_width; //設定inputTarget之寬度
int acc_thr; //擷取累積值閥值
int error_template_times;//matchtemplate錯誤次數，超過即尋找
int testPicture; //更改辨識模式，偵測單張結果
std::string testPicture_path; //測試圖片之路徑
std::string testPicture_savepath; //測試圖片結果存檔路徑
//config.ini

cv::Mat copyFrame;

std::string Xray_result0 = Xray_result0 + "{" + '"' + "rect" + '"' + ":[]}";
int deepColorX, deepColorY, deepColorW, deepColorH;

AutoPtr<IniFileConfiguration> pConf(new IniFileConfiguration("config.ini"));
std::string cfg = pConf->getString("YOLO.cfg");
std::string weights = pConf->getString("YOLO.weights");
std::string names = pConf->getString("YOLO.names");

std::string cfg2 = pConf->getString("YOLO.cfg2");
std::string weights2 = pConf->getString("YOLO.weights2");
std::string names2 = pConf->getString("YOLO.names2");

Detector detector(cfg, weights);
Detector detector2(cfg2, weights2);

Session* session;
int matchtemplate_error_times;//matchtemplate錯誤次數，超過即尋找

void deepColorDetect(cv::Mat mat_img) {
	deepColorX = 0;
	deepColorY = 0;
	deepColorW = 0;
	deepColorH = 0;

	cv::Mat ori, threshold_img, dst, gray;
	ori = mat_img;
	//clock_t a, b;
	int morph_size = 6;
	int morph_elem = 0;
	int operation = 2;
	int bin_thr = 60;
	cv::Mat element = cv::getStructuringElement(morph_elem, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
	std::cout << "****************  getStructuringElement after" << std::endl;
	std::vector<float>contour_area;
	std::vector<std::vector<cv::Point> > contours, largest_contours;
	//a = clock();
	std::cout << "ori.empty: " << ori.empty() << std::endl;
	cv::cvtColor(ori, gray, cv::COLOR_BGR2GRAY);
	std::cout << "****************  cv::cvtColor after" << std::endl;
	threshold(gray, threshold_img, bin_thr, 255, cv::THRESH_BINARY_INV);
	std::cout << "****************  threshold after" << std::endl;

	morphologyEx(threshold_img, dst, operation, element);
	std::cout << "****************  morphologyEx after" << std::endl;
	findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	std::cout << "****************  findContours after" << std::endl;
	double largest_area = 0;
	double area;

	//找最大的輪廓
	for (int i = 0; i < contours.size(); i++) {  // get the largest contour
		area = (contourArea(contours[i]));
		if (area >= largest_area) {
			largest_area = area;
			largest_contours.clear();
			largest_contours.push_back(contours[i]);
		}
	}
	if (debugMode)
		//drawContours(ori, largest_contours, -1, cv::Scalar(0, 0, 255), 2);
	std::cout << "****************  contourArea after" << std::endl;


	if (largest_contours.size() > 0)
	{
		std::vector<cv::Rect>boundRect(largest_contours.size());
		boundRect[0] = boundingRect(largest_contours[0]);
		//rectangle(ori, boundRect[0], cv::Scalar(0, 0, 255), 2);
		std::cout << "****************  rectangle after" << std::endl;
		cv::Point tl, br;
		tl = boundRect[0].tl();
		br = boundRect[0].br();

		contour_area.push_back(largest_area);

		if (largest_area > largest_area_thr)
		{
			
			std::cout << "有大面積深色" << std::endl;
			deepColorX = tl.x;
			deepColorY = tl.y;
			deepColorW = br.x - tl.x;
			deepColorH = br.y - tl.y;
			deepColorResult = deepColorResult + "{" + '"' + "rect" + '"' + ":[" + "{" + '"' + "type" + '"' + ":" + '"' + std::to_string(13) +
				'"' + "," + '"' + "width" + '"' + ":" + '"' + std::to_string(deepColorW) +
				'"' + "," + '"' + "height" + '"' + ":" + '"' + std::to_string(deepColorH) +
				'"' + "," + '"' + "top" + '"' + ":" + '"' + std::to_string(deepColorY) +
				'"' + "," + '"' + "left" + '"' + ":" + '"' + std::to_string(deepColorX) +
				'"' + "}]}";

			//將Equalization 結果覆蓋原圖
			cv::Mat dst;
			dst = mat_img(cv::Rect(deepColorX, deepColorY, deepColorW, deepColorH));
			cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);
			cv::equalizeHist(dst, dst);
			cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);
			
			cv::Rect roi_rect= cv::Rect(deepColorX, deepColorY, deepColorW, deepColorH);
			dst.copyTo(mat_img(roi_rect));
			cv::imwrite(imagePath, mat_img);

			if (debugMode) std::cout << "deepColorResult:" << deepColorResult << std::endl;
		}
		else
		{
			std::cout << "無大面積深色" << std::endl;

		}
	}
	else std::cout << "無大面積深色" << std::endl;

	if (debugMode)
	{
		//imshow("ori", ori);
		//cv::waitKey(1);
	}

}

//將偵測出ROI依x軸位置排序
std::vector<bbox_t> vectorSort(std::vector<bbox_t> sortVector)
{
	int N = sortVector.size();
	for (int i = 0; i < N; i++)
		for (int j = i + 1; j < N; j++)
		{
			if (sortVector[i].x > sortVector[j].x)
				std::swap(sortVector[i], sortVector[j]);//SWAP
		}
	return sortVector;
}

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<bbox_t> result_vec2, std::vector<std::string> obj_names, std::vector<std::string> obj_names2)
{
	if (debugMode)
	{
		for (/*auto& i : result_vec*/int j = 0; j < result_vec.size(); j++) {
			bbox_t i = result_vec[j];
			cv::Scalar color = obj_id_to_color(i.obj_id);
			cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
			if ((obj_names.size() > i.obj_id)) {
				std::string obj_name = obj_names[i.obj_id];
				std::string prob = std::to_string(i.prob).substr(0, std::to_string(i.prob).length() - 3);
				if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
				cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
				int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
				max_width = std::max(max_width, (int)i.w + 2);
				//max_width = std::max(max_width, 283);
				std::string coords_3d;
				if (!std::isnan(i.z_3d)) {
					std::stringstream ss;
					ss << std::fixed << std::setprecision(2) << "x:" << i.x_3d << "m y:" << i.y_3d << "m z:" << i.z_3d << "m ";
					coords_3d = ss.str();
					cv::Size const text_size_3d = getTextSize(ss.str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1, 0);
					int const max_width_3d = (text_size_3d.width > i.w + 2) ? text_size_3d.width : (i.w + 2);
					if (max_width_3d > max_width) max_width = max_width_3d;
				}
				cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
					cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
					color, CV_FILLED, 8, 0);
				putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
				putText(mat_img, prob, cv::Point2f(i.x, i.y + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
				if (!coords_3d.empty()) putText(mat_img, coords_3d, cv::Point2f(i.x, i.y - 1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);
				
				//testsaveImage = test_Path + image_Name;
				//cv::imwrite(testsaveImage, mat_img);
				std::cout << "i.prob : " << i.prob << std::endl;
				//imshow("img", mat_img);
				//cv::waitKey(1);
			}
		}
		for (/*auto& i : result_vec2*/int j = 0; j < result_vec2.size(); j++) {
			bbox_t i = result_vec2[j];
			cv::Scalar color = obj_id_to_color(i.obj_id);
			cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
			if ((obj_names2.size() > i.obj_id)) {
				std::string obj_name = obj_names2[i.obj_id];
				std::string prob = std::to_string(i.prob).substr(0, std::to_string(i.prob).length() - 3);
				if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
				cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
				int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
				max_width = std::max(max_width, (int)i.w + 2);
				//max_width = std::max(max_width, 283);
				std::string coords_3d;
				if (!std::isnan(i.z_3d)) {
					std::stringstream ss;
					ss << std::fixed << std::setprecision(2) << "x:" << i.x_3d << "m y:" << i.y_3d << "m z:" << i.z_3d << "m ";
					coords_3d = ss.str();
					cv::Size const text_size_3d = getTextSize(ss.str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1, 0);
					int const max_width_3d = (text_size_3d.width > i.w + 2) ? text_size_3d.width : (i.w + 2);
					if (max_width_3d > max_width) max_width = max_width_3d;
				}
				cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
					cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
					color, CV_FILLED, 8, 0);
				putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
				putText(mat_img, prob, cv::Point2f(i.x, i.y + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
				if (!coords_3d.empty()) putText(mat_img, coords_3d, cv::Point2f(i.x, i.y - 1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);
				//cv::imwrite(testsaveImage, mat_img);
				std::cout << "i.prob : " << i.prob << std::endl;
				//imshow("img", mat_img);
				//cv::waitKey(1);
			}
		}
		time_t rawtime;
		struct tm* timeinfo;
		char char_fileDate[50], folderDate[50];
		std::string filename;
		time(&rawtime);
		timeinfo = localtime(&rawtime);
		strftime(char_fileDate, 50, "%Y%m%d%H%M-%S", timeinfo);
		strftime(folderDate, 10, "%Y%m%d", timeinfo);

		std::string str_fileDate(char_fileDate);
		std::stringstream ss;
		//儲存辨識影像
		std::string imagePath(folderDate);
		imagePath = test_Path + imagePath;
		//imagePath = "D:\\output\\" + imagePath;
		char carImagePath_char[30];
		strcpy(carImagePath_char, imagePath.c_str());
		struct stat buf;		//檢查是否有當日的資料夾，沒有則新增
		if (stat(carImagePath_char, &buf) != 0)
		{
			mkdir(carImagePath_char);
		}

		picfilename_single = str_fileDate + ".jpg";
		ss << imagePath << "\\" << picfilename_single;
		imagePath = ss.str();
		ss.str("");
		cv::imwrite(imagePath, mat_img);
		//testsaveImage = test_Path + image_Name;
		//cv::imwrite(testsaveImage, mat_img);
	}


	int checknum = 0;
	for (/*auto& i : result_vec*/int j = 0; j < result_vec.size(); j++)
	{
		bbox_t i = result_vec[j];
		if ((i.prob >= thresholdProbClass[i.obj_id]) /* || (i.obj_id == 1 && i.prob >= 0.75) || (i.obj_id == 4 && i.prob >= 0.75)*/)
		{
			checknum++;
			if (checknum == 1)
			{
				Xray_result = Xray_result + "{" + '"' + "rect" + '"' + ":[" + "{" + '"' + "type" + '"' + ":" + '"' + std::to_string(i.obj_id) +
					'"' + "," + '"' + "width" + '"' + ":" + '"' + std::to_string(i.w) +
					'"' + "," + '"' + "height" + '"' + ":" + '"' + std::to_string(i.h) +
					'"' + "," + '"' + "top" + '"' + ":" + '"' + std::to_string(i.y) +
					'"' + "," + '"' + "left" + '"' + ":" + '"' + std::to_string(i.x) +
					'"' + "}";
			}
			else if (checknum >= 2)
			{
				Xray_result = Xray_result + "," + "{" + '"' + "type" + '"' + ":" + '"' + std::to_string(i.obj_id) +
					'"' + "," + '"' + "width" + '"' + ":" + '"' + std::to_string(i.w) +
					'"' + "," + '"' + "height" + '"' + ":" + '"' + std::to_string(i.h) +
					'"' + "," + '"' + "top" + '"' + ":" + '"' + std::to_string(i.y) +
					'"' + "," + '"' + "left" + '"' + ":" + '"' + std::to_string(i.x) +
					'"' + "}";
			}
		}
	}
	for (/*auto& i : result_vec2*/int j = 0; j < result_vec2.size(); j++)
	{
		bbox_t i = result_vec2[j];
		if (i.prob >= thresholdProb2)
		{
			checknum++;
			//Sausage id:4
			if (checknum == 1)
			{
				Xray_result = Xray_result + "{" + '"' + "rect" + '"' + ":[" + "{" + '"' + "type" + '"' + ":" + '"' + std::to_string(7) +
					'"' + "," + '"' + "width" + '"' + ":" + '"' + std::to_string(i.w) +
					'"' + "," + '"' + "height" + '"' + ":" + '"' + std::to_string(i.h) +
					'"' + "," + '"' + "top" + '"' + ":" + '"' + std::to_string(i.y) +
					'"' + "," + '"' + "left" + '"' + ":" + '"' + std::to_string(i.x) +
					'"' + "}";
			}
			else if (checknum >= 2)
			{
				Xray_result = Xray_result + "," + "{" + '"' + "type" + '"' + ":" + '"' + std::to_string(7) +
					'"' + "," + '"' + "width" + '"' + ":" + '"' + std::to_string(i.w) +
					'"' + "," + '"' + "height" + '"' + ":" + '"' + std::to_string(i.h) +
					'"' + "," + '"' + "top" + '"' + ":" + '"' + std::to_string(i.y) +
					'"' + "," + '"' + "left" + '"' + ":" + '"' + std::to_string(i.x) +
					'"' + "}";
			}
		}
	}
	if (debugMode)
		std::cout << "Detect num: " << checknum << std::endl;
	if (checknum > 0)
		Xray_result = Xray_result + "]}";
	else
		Xray_result = Xray_result0;

	checknum = 0;
	std::cout << Xray_result << std::endl;

}

std::vector<std::string> objects_names_from_file(std::string const filename) {
	std::ifstream file(filename);
	std::vector<std::string> file_lines;
	if (!file.is_open()) return file_lines;
	for (std::string line; getline(file, line);) file_lines.push_back(line);
	//std::cout << "object names loaded \n";
	return file_lines;
}

int getAccument_new(int origin_col, int mode) //根據mode 0 or 1 分別從origin_col，向後或向前查找最低像素累積值並return。
{
	int s[1920]; //初始陣列 ，有變數無法帶入的問題。
	int len = sizeof(s) / sizeof(s[0]); //找陣列長度
	int temp;
	int smallest_num = 9999;
	int index_col;

	if (mode == 0) {
		for (int i = 0; i < len; i++) //初始陣列
			s[i] = 9999;
		for (int i = origin_col; i < 1920; i++) {
			cv::Mat A;
			A = bin.col(i);
			s[i] = sum(A)[0] / 255;

			temp = sum(A)[0] / 255;
			if (temp < smallest_num) {
				smallest_num = temp;
				index_col = i;
			}
		}

	}
	else {
		for (int i = 0; i < len; i++) //初始陣列
			s[i] = 9999;
		for (int i = origin_col; i > 0; i--) {
			cv::Mat A;
			A = bin.col(i);
			s[i] = sum(A)[0] / 255;
			temp = sum(A)[0] / 255;
			if (temp < smallest_num) {
				smallest_num = temp;
				index_col = i;
			}
		}
	}
	// print 最小之累積量
	//std::cout << smallest_num << endl;
	return index_col;
}

void takePicture_double(cv::Mat cameraImage, cv::Mat cameraImage1, std::string folderpath)
{
	time_t rawtime;
	struct tm* timeinfo;
	char char_fileDate[50], folderDate[50];
	std::string filename;
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strftime(char_fileDate, 50, "%Y%m%d%H%M-%S", timeinfo);
	strftime(folderDate, 10, "%Y%m%d", timeinfo);
	std::string str_fileDate(char_fileDate);
	std::stringstream ss;
	std::stringstream ss1;
	//儲存辨識影像
	std::string imagePath(folderDate);
	std::string imagePath1;
	imagePath = folderpath + imagePath;
	imagePath1 = imagePath;
	char carImagePath_char[30];
	strcpy(carImagePath_char, imagePath.c_str());
	struct stat buf;		//檢查是否有當日的資料夾，沒有則新增
	if (stat(carImagePath_char, &buf) != 0)
	{
		mkdir(carImagePath_char);
	}
	picfilename_double1 = str_fileDate + "_A.jpg";
	picfilename_double2 = str_fileDate + "_B.jpg";
	ss << imagePath << "\\" << picfilename_double1;
	ss1 << imagePath << "\\" << picfilename_double2;
	imagePath = ss.str();
	imagePath1 = ss1.str();

	imwrite(imagePath, cameraImage);
	imwrite(imagePath1, cameraImage1);
	imagePath_double1 = imagePath;
	imagePath_double2 = imagePath1;

}

void takePicture_single(cv::Mat cameraImage, std::string folderpath)
{
	time_t rawtime;
	struct tm* timeinfo;
	char char_fileDate[50], folderDate[50];
	std::string filename;
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strftime(char_fileDate, 50, "%Y%m%d%H%M-%S", timeinfo);
	strftime(folderDate, 10, "%Y%m%d", timeinfo);

	std::string str_fileDate(char_fileDate);
	std::stringstream ss;
	//儲存辨識影像
	std::string imagePath(folderDate);
	imagePath = folderpath + imagePath;
	//imagePath = "D:\\output\\" + imagePath;
	char carImagePath_char[30];
	strcpy(carImagePath_char, imagePath.c_str());
	struct stat buf;		//檢查是否有當日的資料夾，沒有則新增
	if (stat(carImagePath_char, &buf) != 0)
	{
		mkdir(carImagePath_char);
	}
	picfilename_single = str_fileDate + ".jpg";
	ss << imagePath << "\\" << picfilename_single;
	imagePath = ss.str();
	ss.str("");
	imwrite(imagePath, cameraImage);
	imagePath_single = imagePath;

}

void XrayAicheck_process(cv::Mat frame) {
	std::vector<std::string> obj_names = objects_names_from_file(names);
	std::vector<std::string> obj_names2 = objects_names_from_file(names2);

	std::string method2_Check = "0";
	std::cout << std::endl << std::endl << saveFilename << std::endl;
	cv::waitKey(1);
	while (saveFilename.find("/") != std::string::npos)
	{
		saveFilename = saveFilename.replace(saveFilename.find("/"), 1, "\\");
	}

	try {
		*session << "INSERT INTO imgInfoView (imgName,imgPath,xrayType) VALUES(?,?,?)",
			Poco::Data::Keywords::use(image_Name),
			Poco::Data::Keywords::use(imagePath),
			Poco::Data::Keywords::use(xraytype),
			Poco::Data::Keywords::now;

	}
	catch (std::exception& e)
	{
		std::cerr << "insert imgInfo exception: " << e.what() << "\n";
	}
	//inputIMG
	//frame = cv::imread(saveFilename);
	frame.copyTo(copyFrame);

	deepColorDetect(copyFrame);
	std::cout << "=== deepColorDetect after" << std::endl;
	std::vector<bbox_t> result_CN;
	std::vector<bbox_t> result_CN2;

	result_CN = detector.detect(frame);
	result_CN2 = detector2.detect(frame);
	std::cout << "=== detector after" << std::endl;
	result_CN = vectorSort(result_CN);
	result_CN2 = vectorSort(result_CN2);

	if (result_CN.size() != 0)
	{
		int count = 0;
		draw_boxes(frame, result_CN, result_CN2, obj_names, obj_names2);
	}
	else
	{
		Xray_result = Xray_result0;
	}

	if (Xray_result == Xray_result0)
	{
		method2_Check = "0";
	}
	else
	{
		method2_Check = "1";
	}
	std::cout << "=== draw_boxes after" << std::endl;
	try
	{
		*session << "INSERT INTO algorithmResultView (imgName,mType,mResult,mStatus) VALUES(?,1,?,2)",
			Poco::Data::Keywords::use(image_Name),
			Poco::Data::Keywords::use(Xray_result),
			Poco::Data::Keywords::now;
	}
	catch (std::exception& e)
	{
		std::cerr << "insert algorithmResult exception: " << e.what() << "\n";
	}



	if (deepColorW != 0)
	{
		try {
			*session << "INSERT INTO algorithmResultView (imgName,mType,mResult,mStatus) VALUES(?,3,?,2)",
				Poco::Data::Keywords::use(image_Name),
				Poco::Data::Keywords::use(deepColorResult),
				Poco::Data::Keywords::now;
		}
		catch (std::exception& e)
		{
			std::cerr << "insert deep color exception: " << e.what() << "\n";
		}
	}
	inputFilename.clear();
	Xray_result.clear();
	deepColorResult.clear();
}

void draw_boxes_testPicture(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<bbox_t> result_vec2, std::vector<std::string> obj_names, std::vector<std::string> obj_names2)
{
	for (int j = 0; j < result_vec.size(); j++) {
		bbox_t i = result_vec[j];
		cv::Scalar color = obj_id_to_color(i.obj_id);
		cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
		if ((obj_names.size() > i.obj_id)) {
			std::string obj_name = obj_names[i.obj_id];
			std::string prob = std::to_string(i.prob).substr(0, std::to_string(i.prob).length() - 3);
			if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
			cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
			int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
			max_width = std::max(max_width, (int)i.w + 2);
			std::string coords_3d;
			if (!std::isnan(i.z_3d)) {
				std::stringstream ss;
				ss << std::fixed << std::setprecision(2) << "x:" << i.x_3d << "m y:" << i.y_3d << "m z:" << i.z_3d << "m ";
				coords_3d = ss.str();
				cv::Size const text_size_3d = getTextSize(ss.str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1, 0);
				int const max_width_3d = (text_size_3d.width > i.w + 2) ? text_size_3d.width : (i.w + 2);
				if (max_width_3d > max_width) max_width = max_width_3d;
			}
			cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
				cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
				color, CV_FILLED, 8, 0);
			putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
			putText(mat_img, prob, cv::Point2f(i.x, i.y + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
			if (!coords_3d.empty()) putText(mat_img, coords_3d, cv::Point2f(i.x, i.y - 1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);
			std::cout << "i.prob : " << i.prob << std::endl;
		}
	}
	for (int j = 0; j < result_vec2.size(); j++) {
		bbox_t i = result_vec2[j];
		cv::Scalar color = obj_id_to_color(i.obj_id);
		cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
		if ((obj_names2.size() > i.obj_id)) {
			std::string obj_name = obj_names2[i.obj_id];
			std::string prob = std::to_string(i.prob).substr(0, std::to_string(i.prob).length() - 3);
			if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
			cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
			int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
			max_width = std::max(max_width, (int)i.w + 2);
			std::string coords_3d;
			if (!std::isnan(i.z_3d)) {
				std::stringstream ss;
				ss << std::fixed << std::setprecision(2) << "x:" << i.x_3d << "m y:" << i.y_3d << "m z:" << i.z_3d << "m ";
				coords_3d = ss.str();
				cv::Size const text_size_3d = getTextSize(ss.str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1, 0);
				int const max_width_3d = (text_size_3d.width > i.w + 2) ? text_size_3d.width : (i.w + 2);
				if (max_width_3d > max_width) max_width = max_width_3d;
			}
			cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
				cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
				color, CV_FILLED, 8, 0);
			putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
			putText(mat_img, prob, cv::Point2f(i.x, i.y + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
			if (!coords_3d.empty()) putText(mat_img, coords_3d, cv::Point2f(i.x, i.y - 1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);
			std::cout << "i.prob : " << i.prob << std::endl;
		}
	}

	int checknum = 0;
	for (int j = 0; j < result_vec.size(); j++)
	{
		bbox_t i = result_vec[j];
		if ((i.prob >= thresholdProbClass[i.obj_id]))
		{
			checknum++;
			if (checknum == 1)
			{
				Xray_result = Xray_result + "{" + '"' + "rect" + '"' + ":[" + "{" + '"' + "type" + '"' + ":" + '"' + std::to_string(i.obj_id) +
					'"' + "," + '"' + "width" + '"' + ":" + '"' + std::to_string(i.w) +
					'"' + "," + '"' + "height" + '"' + ":" + '"' + std::to_string(i.h) +
					'"' + "," + '"' + "top" + '"' + ":" + '"' + std::to_string(i.y) +
					'"' + "," + '"' + "left" + '"' + ":" + '"' + std::to_string(i.x) +
					'"' + "}";
			}
			else if (checknum >= 2)
			{
				Xray_result = Xray_result + "," + "{" + '"' + "type" + '"' + ":" + '"' + std::to_string(i.obj_id) +
					'"' + "," + '"' + "width" + '"' + ":" + '"' + std::to_string(i.w) +
					'"' + "," + '"' + "height" + '"' + ":" + '"' + std::to_string(i.h) +
					'"' + "," + '"' + "top" + '"' + ":" + '"' + std::to_string(i.y) +
					'"' + "," + '"' + "left" + '"' + ":" + '"' + std::to_string(i.x) +
					'"' + "}";
			}
		}
	}
	for (int j = 0; j < result_vec2.size(); j++)
	{
		bbox_t i = result_vec2[j];
		if (i.prob >= thresholdProb2)
		{
			checknum++;
			//Sausage id:4
			if (checknum == 1)
			{
				Xray_result = Xray_result + "{" + '"' + "rect" + '"' + ":[" + "{" + '"' + "type" + '"' + ":" + '"' + std::to_string(7) +
					'"' + "," + '"' + "width" + '"' + ":" + '"' + std::to_string(i.w) +
					'"' + "," + '"' + "height" + '"' + ":" + '"' + std::to_string(i.h) +
					'"' + "," + '"' + "top" + '"' + ":" + '"' + std::to_string(i.y) +
					'"' + "," + '"' + "left" + '"' + ":" + '"' + std::to_string(i.x) +
					'"' + "}";
			}
			else if (checknum >= 2)
			{
				Xray_result = Xray_result + "," + "{" + '"' + "type" + '"' + ":" + '"' + std::to_string(7) +
					'"' + "," + '"' + "width" + '"' + ":" + '"' + std::to_string(i.w) +
					'"' + "," + '"' + "height" + '"' + ":" + '"' + std::to_string(i.h) +
					'"' + "," + '"' + "top" + '"' + ":" + '"' + std::to_string(i.y) +
					'"' + "," + '"' + "left" + '"' + ":" + '"' + std::to_string(i.x) +
					'"' + "}";
			}
		}
	}
	if (debugMode)
		std::cout << "Detect num: " << checknum << std::endl;
	if (checknum > 0)
		Xray_result = Xray_result + "]}";
	else
		Xray_result = Xray_result0;

	checknum = 0;
	std::cout << Xray_result << std::endl;
}

void XrayAicheck_process_testPicture(cv::Mat frame) {
	std::vector<std::string> obj_names = objects_names_from_file(names);
	std::vector<std::string> obj_names2 = objects_names_from_file(names2);
	std::string method2_Check = "0";
	frame.copyTo(copyFrame);
	deepColorDetect(copyFrame);
	std::cout << "=== deepColorDetect after" << std::endl;
	std::vector<bbox_t> result_CN;
	std::vector<bbox_t> result_CN2;

	result_CN = detector.detect(frame);
	result_CN2 = detector2.detect(frame);
	std::cout << "=== detector after" << std::endl;
	result_CN = vectorSort(result_CN);
	result_CN2 = vectorSort(result_CN2);

	if (result_CN.size() != 0)
	{
		int count = 0;
		draw_boxes_testPicture(frame, result_CN, result_CN2, obj_names, obj_names2);
	}
}

void loadConfig_test(AutoPtr<IniFileConfiguration> pConf) {
	//try {
	//	dbString = pConf->getString("Main.dbString");
	//	thresholdProb = pConf->getDouble("Main.thresholdProb");

	//	for (int i = 0; i < 14; i++)
	//	{
	//		thresholdProbClass[i] = pConf->getDouble("Main.thresholdProbClass" + std::to_string(i + 1));
	//	}

	//	thresholdProb2 = pConf->getDouble("Main.thresholdProb2");
	//	largest_area_thr = pConf->getDouble("Main.largest_area_thr");

	//	cfg = pConf->getString("YOLO.cfg");
	//	weights = pConf->getString("YOLO.weights");
	//	names = pConf->getString("YOLO.names");

	//	cfg2 = pConf->getString("YOLO.cfg2");
	//	weights2 = pConf->getString("YOLO.weights2");
	//	names2 = pConf->getString("YOLO.names2");

	//	debugMode = pConf->getDouble("Test.debugMode");
	//	test_Path = pConf->getString("Test.test_Path");
	//	if (!std::filesystem::exists(test_Path.c_str()))
	//		std::filesystem::create_directory(test_Path); //若無存在則創立目錄
	//	testPicture = pConf->getInt("Test.testPicture");
	//	testPicture_path = pConf->getString("Test.testPicture_path");
	//	testPicture_savepath = pConf->getString("Test.testPicture_savepath");
	//	if (!std::filesystem::exists(testPicture_savepath.c_str()))
	//		std::filesystem::create_directory(testPicture_savepath);//若無存在則創立目錄

	//	capture0_state = pConf->getString("FrameCapture.capture0_state");
	//	capture1_state = pConf->getString("FrameCapture.capture1_state");
	//	LtoR = pConf->getString("FrameCapture.LtoR");
	//	folder_path = pConf->getString("FrameCapture.folder_path");
	//	duration = pConf->getInt("FrameCapture.duration");
	//	xraytype = pConf->getInt("FrameCapture.xraytype");
	//	bottom_limit = pConf->getInt("FrameCapture.bottom_limit");
	//	thr_line = pConf->getInt("FrameCapture.thr_line");
	//	showpic = pConf->getInt("FrameCapture.showpic");
	//	video_col = pConf->getInt("FrameCapture.video_col");
	//	video_row = pConf->getInt("FrameCapture.video_row");
	//	maxVal_thr = pConf->getDouble("FrameCapture.maxVal_thr");
	//	min_dist = pConf->getInt("FrameCapture.min_dist");
	//	capture_min_time = pConf->getDouble("FrameCapture.capture_min_time");
	//	frame_height = pConf->getInt("FrameCapture.frame_height");
	//	frame_width = pConf->getInt("FrameCapture.frame_width");
	//	frame_coordinate_x = pConf->getInt("FrameCapture.frame_coordinate_x");
	//	frame_coordinate_y = pConf->getInt("FrameCapture.frame_coordinate_y");
	//	inputTarget_width = pConf->getInt("FrameCapture.inputTarget_width");
	//	acc_thr = pConf->getInt("FrameCapture.acc_thr");
	//	error_template_times = pConf->getInt("FrameCapture.error_template_times");
	//	
	//}
	//catch (std::exception& e)
	//{
	//	std::cerr << "Read config.ini fail" << "\n";
	//}
	//if (debugMode)
	//{
	//	std::cout << "Main.dbString:\t\t" << dbString << std::endl;
	//	std::cout << "Main.thresholdProb:\t" << thresholdProb << std::endl;
	//	for (int i = 0; i < 14; i++)
	//	{
	//		std::cout << "Main.thresholdProbClass" << (i + 1) << ":\t" << thresholdProbClass[i] << std::endl;
	//	}
	//	std::cout << "Main.thresholdProb2:\t" << thresholdProb2 << std::endl;
	//	std::cout << "YOLO.cfg:\t\t" << cfg << std::endl;
	//	std::cout << "YOLO.weights:\t\t" << weights << std::endl;
	//	std::cout << "YOLO.names:\t\t" << names << std::endl;
	//	std::cout << "Test.debugMode:\t\t" << debugMode << std::endl;
	//	std::cout << "Test.test_Path:\t\t" << test_Path << std::endl;
	//	std::cout << "Test.testPicture\t\t" << testPicture << std::endl;
	//	std::cout << "Test.testPicture_path\t\t" << testPicture_path << std::endl;
	//	std::cout << "Test.testPicture_savepath\t\t" << testPicture_savepath << std::endl;

	//	std::cout << "capture0_state:" << capture0_state << std::endl;
	//	std::cout << "capture1_state:" << capture1_state << std::endl;
	//	std::cout << "LtoR:" << LtoR << std::endl; // 0運輸帶右向左 or 1運輸帶左向右
	//	std::cout << "folder_path:" << folder_path << std::endl; // 儲存資料夾
	//	std::cout << "duration:" << duration << std::endl; // 運作幀數
	//	std::cout << "xraytype:" << xraytype << std::endl; // 0單光源 or 1雙光源
	//	std::cout << "bottom_limit:" << bottom_limit << std::endl; // 分割畫面(?列以下皆忽略)
	//	std::cout << "thr_line:" << thr_line << std::endl; // 中央基準線
	//	std::cout << "showpic:" << showpic << std::endl; // Debug模式
	//	std::cout << "video_col:" << video_col << std::endl;
	//	std::cout << "video_row:" << video_row << std::endl;
	//	std::cout << "maxVal_thr:" << maxVal_thr << std::endl; // matchTemplate定位閥值
	//	std::cout << "min_dist:" << min_dist << std::endl; // 擷取之物件最小距離
	//	std::cout << "capture_min_time:" << capture_min_time << std::endl; // 擷取時間最短限制
	//	std::cout << "frame_height & width:" << frame_height << " x " << frame_width << std::endl; // 擷取時間最短限制
	//	std::cout << "frame_coordinate_x & y:" << " (" << frame_coordinate_x << "," << frame_coordinate_y << ") " << std::endl; // 擷取時間最短限制
	//	std::cout << "intputTarget_width:" << inputTarget_width << std::endl; //intputTarget之寬度
	//	std::cout << "acc_thr:" << acc_thr << std::endl; //擷取累積值閥值
	//	std::cout << "error_template_times:" << error_template_times << std::endl; ////matchtemplate錯誤次數，超過即尋找
	//}
}

void testPicture_fun() {
	//for (const auto& entry : std::filesystem::directory_iterator(testPicture_path)) {
	//	//std::filesystem::directory_entry entry;
	//	
	//	std::cout << entry.path();
	//	cv::Mat img = cv::imread(entry.path().string());
	//	XrayAicheck_process_testPicture(img);
	//	//cv::imshow("img", img);
	//	std::string name = entry.path().filename().string();
	//	std::string path = testPicture_savepath + name;
	//	cv::imwrite(path, img);
	//}
}

int main(int argc, char* argv[])
{
	//XrayAiCheck 初始化
	AutoPtr<IniFileConfiguration> pConf(new IniFileConfiguration("config.ini"));
	loadConfig_test(pConf);

	//創建一空圖檔輸入YOLO辨識，因第一次辨識較耗時
	cv::Mat mat_Default = cv::Mat(1, 1, CV_8UC3);
	std::vector<bbox_t> result_CN = detector.detect(mat_Default);
	//std::vector<bbox_t> result_CN2 = detector2.detect(mat_Default);

	Poco::Data::ODBC::Connector::registerConnector();
	const std::string& connString = dbString;
	
	try {
		session = new Session("ODBC", connString);
	}
	catch (std::exception& e)
	{
		std::cerr << "資料庫連線失敗 " << e.what() << "\n";
	}
	int j = 0;

	double START, END; START = clock();
	double capture_start, capture_end; capture_start = clock();
	
	if (testPicture == 1) {
		std::cout << "單張測試" << std::endl;
		testPicture_fun();
		return 0;
	}

	//測試影片輸入
	cv::VideoCapture capture, capture1;
	if (capture0_state.compare("0") == 0)
		capture.open(stoi(capture0_state));

	else {
		if ((capture0_state != "0") || (capture0_state != "1")) {//測試輸入影片檔名
			capture.open(capture0_state);
		}
	}


	if (capture1_state.compare("1") == 0)
		capture1.open(stoi(capture1_state));
	else {

		if ((capture1_state != "0") || (capture1_state != "1")) { //測試輸入影片檔名
			capture1.open(capture1_state);
		}
	}

	if (!capture.isOpened())
	{
		std::cout << "Read video Failed !" << std::endl;
		return 0;
	}

	if (!capture1.isOpened())
	{
		std::cout << "Read video1 Failed !" << std::endl;
		return 0;
	}



	if (showpic != 0) {
		if(xraytype==0){		
			cv::namedWindow("frame", CV_WINDOW_NORMAL);
			cv::resizeWindow("frame", frame_width-17, frame_height-39); //存在偏差 與實際設定不相符
			cv::moveWindow("frame", frame_coordinate_x, frame_coordinate_y);
			cv::setWindowProperty("frame", cv::WND_PROP_TOPMOST, 1);
			//將cv::imshow之titlebar消除。
			// Get window handle. It will return the handle of the video container
			// There is also enclosing parent window, which will be treated later
			HWND m_hMediaWindow = (HWND)cvGetWindowHandle("frame");
			// change style of the child HighGui window
			DWORD style = ::GetWindowLong(m_hMediaWindow, GWL_STYLE);
			style &= ~WS_OVERLAPPEDWINDOW;
			style |= WS_POPUP;
			::SetWindowLong(m_hMediaWindow, GWL_STYLE, style);
			// change style of the parent HighGui window
			HWND hParent = ::FindWindow(0, CString("frame"));
			style = ::GetWindowLong(hParent, GWL_STYLE);
			style &= ~WS_OVERLAPPEDWINDOW;
			style |= WS_POPUP;
			::SetWindowLong(hParent, GWL_STYLE, style);
		}
		else {
			cv::namedWindow("frame", CV_WINDOW_NORMAL); 
			cv::namedWindow("frame1", CV_WINDOW_NORMAL);
			cv::resizeWindow("frame", frame_width - 17, frame_height - 39); //存在偏差 與實際設定不相符
			cv::resizeWindow("frame1", frame_width - 17, frame_height - 39); //存在偏差 與實際設定不相符
			cv::moveWindow("frame", frame_coordinate_x, frame_coordinate_y);
			cv::moveWindow("frame1", frame_coordinate_x+946, frame_coordinate_y); 
			cv::setWindowProperty("frame", cv::WND_PROP_TOPMOST, 1);
			cv::setWindowProperty("frame1", cv::WND_PROP_TOPMOST, 1);
			// Get window handle. It will return the handle of the video container
			// There is also enclosing parent window, which will be treated later
			HWND m_hMediaWindow = (HWND)cvGetWindowHandle("frame");
			HWND m_hMediaWindow1 = (HWND)cvGetWindowHandle("frame1");
			// change style of the child HighGui window
			DWORD style = ::GetWindowLong(m_hMediaWindow, GWL_STYLE);
			DWORD style1 = ::GetWindowLong(m_hMediaWindow1, GWL_STYLE);
			style &= ~WS_OVERLAPPEDWINDOW;
			style |= WS_POPUP;
			::SetWindowLong(m_hMediaWindow, GWL_STYLE, style);
			::SetWindowLong(m_hMediaWindow1, GWL_STYLE, style);
			// change style of the parent HighGui window
			HWND hParent = ::FindWindow(0, CString("frame"));
			HWND hParent1 = ::FindWindow(0, CString("frame1"));
			style = ::GetWindowLong(hParent, GWL_STYLE);
			style = ::GetWindowLong(hParent1, GWL_STYLE);
			style &= ~WS_OVERLAPPEDWINDOW;
			style |= WS_POPUP;
			::SetWindowLong(hParent, GWL_STYLE, style);
			::SetWindowLong(hParent1, GWL_STYLE, style);
		}
		
	}
	capture.set(cv::CAP_PROP_FRAME_WIDTH, video_col);
	capture.set(cv::CAP_PROP_FRAME_HEIGHT, video_row);
	capture1.set(cv::CAP_PROP_FRAME_WIDTH, video_col);
	capture1.set(cv::CAP_PROP_FRAME_HEIGHT, video_row);

	int frame_num = 0;
	int count = 0;

	if (xraytype == 1)//雙射源
	{

		while (true)
		{

			//畫基準判斷線供使用者參考 並塗抹X光機下方工具列
			capture >> frame;
			capture1 >> frame1;

			if (frame.empty() || frame1.empty()) {
				break;
			}

			frame = frame(cv::Range(0, bottom_limit), cv::Range(0, video_col));//CaptureRoi(cv::Range(row),cv::Range(col))
			frame1 = frame1(cv::Range(0, bottom_limit), cv::Range(0, video_col));//CaptureRoi(cv::Range(row),cv::Range(col))

			if (frame_num % duration == 0)
			{
				ones = false;
				cv::cvtColor(frame, roi_start, cv::COLOR_BGR2GRAY);
				cv::Mat origin, origin1;
				frame.copyTo(origin);
				frame1.copyTo(origin1);

				if (showpic != 0) {
					line(frame, cv::Point(thr_line, 0), cv::Point(thr_line, bottom_limit), cv::Scalar(255, 0, 0), 4);
					line(frame1, cv::Point(thr_line, 0), cv::Point(thr_line, bottom_limit), cv::Scalar(255, 0, 0), 4);
				}

				cv::cvtColor(origin, gray, cv::COLOR_BGR2GRAY);
				threshold(gray, bin, 200, 255, cv::THRESH_BINARY_INV);
				if (stoi(LtoR) == 0)//LtoR = 0物件由右至左 <-- 
				{
					if (inputTarget.empty()) //倘使無初始相片可供matchTemplate比對，則自行找
					{
						int col = getAccument_new(thr_line, stoi(LtoR));
						if (showpic != 0) {
							line(frame, cv::Point(col, 0), cv::Point(col, bottom_limit), cv::Scalar(0, 0, 255), 4);
							line(frame1, cv::Point(col, 0), cv::Point(col, bottom_limit), cv::Scalar(0, 0, 255), 4);
						}
						res = origin(cv::Range(0, bottom_limit), cv::Range(thr_line, col));
						res1 = origin1(cv::Range(0, bottom_limit), cv::Range(thr_line, col));
						inputTarget = origin(cv::Range(0, bottom_limit), cv::Range(col - inputTarget_width, col));//CaptureRoi(cv::Range(row),cv::Range(col))
						cv::cvtColor(inputTarget, inputTarget, cv::COLOR_BGR2GRAY);
					}
					/// Create the result matrix
					int result_cols = video_col - inputTarget.cols + 1;
					int result_rows = video_row - inputTarget.rows + 1;
					cv::Mat result(result_cols, result_rows, CV_32FC1);

					matchTemplate(gray, inputTarget, result, cv::TM_CCOEFF_NORMED);
					minMaxLoc(result, &minVal, &maxVal, NULL, &Loc);

					//確認是否全為相同之值，以避免導致matchTemplate異常
					if (maxVal == 1) {
						if (showpic != 0) {
							std::cout << "maxVal=1 比對異常" << std::endl;
						}
						maxVal = 0;
					}
					//std::cout <<"maxVal = "<< maxVal << endl;
					if (maxVal >= maxVal_thr)//根據比對結果框選邊框，並決定擷取條件。
					{
						matchtemplate_error_times = 0;
						rectangle(frame, Loc, cv::Point(Loc.x + inputTarget.cols, Loc.y + inputTarget.rows), cv::Scalar(0, 255, 0), 4);
						rectangle(frame1, Loc, cv::Point(Loc.x + inputTarget.cols, Loc.y + inputTarget.rows), cv::Scalar(0, 255, 0), 4);
						if (video_col - (Loc.x + inputTarget.cols) > min_dist) { //大於一定寬度開始找尋
							cv::Mat A = bin.col(video_col - 10);
							int temp = sum(A)[0] / 255;
							std::cout << "現在累積值 = " << temp << std::endl;
							if (temp < acc_thr) {
								//紀錄時間
								capture_end = clock();
								//截圖時長過短則忽略
								if ((capture_end - capture_start) / CLOCKS_PER_SEC < capture_min_time)
									std::cout << "截圖費時: " << (capture_end - capture_start) / CLOCKS_PER_SEC << ", 時長過短忽略" << std::endl;
								else {
									std::cout << "截圖費時: " << (capture_end - capture_start) / CLOCKS_PER_SEC << std::endl;
									capture_start = capture_end;
									//截圖
									if (showpic != 0) {
										std::cout << "找到最低點截圖" << std::endl;
									}
									res = origin(cv::Range(0, bottom_limit), cv::Range(Loc.x + inputTarget.cols, video_col));
									res1 = origin1(cv::Range(0, bottom_limit), cv::Range(Loc.x + inputTarget.cols, video_col));

									takePicture_double(res, res1, folder_path);
									image_Name = picfilename_double1;
									imagePath = imagePath_double1;
									XrayAicheck_process(res);

									image_Name = picfilename_double2;
									imagePath = imagePath_double2;
									XrayAicheck_process(res1);

									inputTarget = origin(cv::Range(0, bottom_limit), cv::Range(video_col - inputTarget_width, video_col));
									cv::cvtColor(inputTarget, inputTarget, cv::COLOR_BGR2GRAY);

								}

							}
						}
						if ((Loc.x + inputTarget.cols) <= thr_line) { //如果有過長的寬度則截圖
							//紀錄時間
							capture_end = clock();
							//截圖時長過短則忽略
							if ((capture_end - capture_start) / CLOCKS_PER_SEC < capture_min_time)
								std::cout << "截圖費時: " << (capture_end - capture_start) / CLOCKS_PER_SEC << ", 時長過短忽略" << std::endl;
							else {
								std::cout << "截圖費時: " << (capture_end - capture_start) / CLOCKS_PER_SEC << std::endl;
								capture_start = capture_end;
								//截圖
								if (showpic != 0) {
									std::cout << "未找到最低點 但物件過長 向後尋找可能的物件邊界並截圖" << std::endl;
									//std::cout << "未找到最低點但物件過長截圖" << endl;
								}
								// 若物件過長向後找尋最低點位置
								int col = getAccument_new(Loc.x + inputTarget.cols + min_dist, stoi(LtoR));
								// 20220216 向後找尋最低點
								res = origin(cv::Range(0, bottom_limit), cv::Range(Loc.x + inputTarget.cols, col));
								res1 = origin1(cv::Range(0, bottom_limit), cv::Range(Loc.x + inputTarget.cols, col));

								takePicture_double(res, res1, folder_path);
								image_Name = picfilename_double1;
								imagePath = imagePath_double1;
								XrayAicheck_process(res);

								image_Name = picfilename_double2;
								imagePath = imagePath_double2;
								XrayAicheck_process(res1);

								inputTarget = origin(cv::Range(0, bottom_limit), cv::Range(col - inputTarget_width, col));
								cv::cvtColor(inputTarget, inputTarget, cv::COLOR_BGR2GRAY);
							}

						}
					}
					else //遺失邊框 找新框
					{
						matchtemplate_error_times += 1;
						if (showpic != 0) {
							std::cout << "遺失邊框" << std::endl;
						}
						if (matchtemplate_error_times == error_template_times) 
						{
							std::cout << "遺失邊框" << matchtemplate_error_times << "次 找新框" << std::endl;
							int col = getAccument_new(thr_line, stoi(LtoR));
							if (showpic != 0) {
								rectangle(frame, cv::Point(col - inputTarget_width, 0), cv::Point(col, bottom_limit), cv::Scalar(0, 0, 255), 4);
								rectangle(frame1, cv::Point(col - inputTarget_width, 0), cv::Point(col, bottom_limit), cv::Scalar(0, 0, 255), 4);
							}
							inputTarget = origin(cv::Range(0, bottom_limit), cv::Range(col - inputTarget_width, col));//CaptureRoi(cv::Range(row),cv::Range(col))
							cv::cvtColor(inputTarget, inputTarget, cv::COLOR_BGR2GRAY);
							matchtemplate_error_times = 0;
						}
					}
				}
				// LtoR = 1 物件由左至右 --> 
				else
				{
					if (inputTarget.empty())//倘使無初始相片可供matchTemplate比對，則自行找
					{
						int col = getAccument_new(thr_line, stoi(LtoR));
						if (showpic != 0) {
							line(frame, cv::Point(col, 0), cv::Point(col, bottom_limit), cv::Scalar(0, 0, 255), 4);
							line(frame1, cv::Point(col, 0), cv::Point(col, bottom_limit), cv::Scalar(0, 0, 255), 4);
						}
						res = origin(cv::Range(0, bottom_limit), cv::Range(col, thr_line));
						res1 = origin1(cv::Range(0, bottom_limit), cv::Range(col, thr_line));
						inputTarget = origin(cv::Range(0, bottom_limit), cv::Range(col, col + inputTarget_width));//CaptureRoi(cv::Range(row),cv::Range(col))
						cv::cvtColor(inputTarget, inputTarget, cv::COLOR_BGR2GRAY);
					}
					/// Create the result matrix
					int result_cols = video_col - inputTarget.cols + 1;
					int result_rows = video_row - inputTarget.rows + 1;
					cv::Mat result(result_cols, result_rows, CV_32FC1);

					matchTemplate(gray, inputTarget, result, cv::TM_CCOEFF_NORMED);
					minMaxLoc(result, &minVal, &maxVal, NULL, &Loc);

					//確認是否全為相同之值，以避免導致matchTemplate異常	
					if (maxVal == 1) {
						if (showpic != 0) {
							std::cout << "maxVal=1 比對異常" << std::endl;
						}
						maxVal = 0;
					}

					if (maxVal >= maxVal_thr)//根據比對結果框選邊框，並決定擷取條件。
					{
						matchtemplate_error_times = 0;
						rectangle(frame, Loc, cv::Point(Loc.x + inputTarget.cols, Loc.y + inputTarget.rows), cv::Scalar(0, 255, 0), 4);
						rectangle(frame1, Loc, cv::Point(Loc.x + inputTarget.cols, Loc.y + inputTarget.rows), cv::Scalar(0, 255, 0), 4);
						if (Loc.x > min_dist) { //大於一定寬度開始找尋
							cv::Mat A = bin.col(10);
							int temp = sum(A)[0] / 255;
							std::cout << "現在累積值 = " << temp << std::endl;
							if (temp < acc_thr) {
								//紀錄時間
								capture_end = clock();
								//截圖時長過短則忽略
								if ((capture_end - capture_start) / CLOCKS_PER_SEC < capture_min_time)
									std::cout << "截圖費時: " << (capture_end - capture_start) / CLOCKS_PER_SEC << ", 時長過短忽略" << std::endl;
								else {
									std::cout << "截圖費時: " << (capture_end - capture_start) / CLOCKS_PER_SEC << std::endl;
									capture_start = capture_end;
									//截圖
									if (showpic != 0) {
										std::cout << "找到最低點截圖" << std::endl;
									}
									res = origin(cv::Range(0, bottom_limit), cv::Range(0, Loc.x));
									res1 = origin1(cv::Range(0, bottom_limit), cv::Range(0, Loc.x));

									takePicture_double(res, res1, folder_path);
									image_Name = picfilename_double1;
									imagePath = imagePath_double1;
									XrayAicheck_process(res);

									image_Name = picfilename_double2;
									imagePath = imagePath_double2;
									XrayAicheck_process(res1);

									inputTarget = origin(cv::Range(0, bottom_limit), cv::Range(0, inputTarget_width));
									cv::cvtColor(inputTarget, inputTarget, cv::COLOR_BGR2GRAY);
								}
							}
						}
						if (Loc.x >= thr_line) { //如果有過長的寬度則截圖
							//紀錄時間
							capture_end = clock();
							//截圖時長過短則忽略
							if ((capture_end - capture_start) / CLOCKS_PER_SEC < capture_min_time)
								std::cout << "截圖費時: " << (capture_end - capture_start) / CLOCKS_PER_SEC << ", 時長過短忽略" << std::endl;
							else {
								std::cout << "截圖費時: " << (capture_end - capture_start) / CLOCKS_PER_SEC << std::endl;
								capture_start = capture_end;
								//截圖
								if (showpic != 0) {
									std::cout << "未找到最低點 但物件過長 向後尋找可能的物件邊界並截圖" << std::endl;
								}
								// 若物件過長向後找尋最低點位置
								int col = getAccument_new(Loc.x - min_dist, stoi(LtoR));
								// 20220216 向後找尋最低點
								res = origin(cv::Range(0, bottom_limit), cv::Range(col, Loc.x));
								res1 = origin1(cv::Range(0, bottom_limit), cv::Range(col, Loc.x));

								takePicture_double(res, res1, folder_path);
								image_Name = picfilename_double1;
								imagePath = imagePath_double1;
								XrayAicheck_process(res);

								image_Name = picfilename_double2;
								imagePath = imagePath_double2;
								XrayAicheck_process(res1);

								inputTarget = origin(cv::Range(0, bottom_limit), cv::Range(col, col + inputTarget_width));
								cv::cvtColor(inputTarget, inputTarget, cv::COLOR_BGR2GRAY);
							}

						}
					}
					else //遺失邊框 找新框
					{
						matchtemplate_error_times += 1;
						if (showpic != 0) {
							std::cout << "遺失邊框" << std::endl;
						}
						if (matchtemplate_error_times == error_template_times) 
						{
							std::cout << "遺失邊框"<<matchtemplate_error_times <<"次 找新框" << std::endl;
							int col = getAccument_new(thr_line, stoi(LtoR));
							if (showpic != 0) {
								rectangle(frame, cv::Point(col, 0), cv::Point(col + inputTarget_width, bottom_limit), cv::Scalar(0, 0, 255), 4);
								rectangle(frame1, cv::Point(col, 0), cv::Point(col + inputTarget_width, bottom_limit), cv::Scalar(0, 0, 255), 4);
							}
							inputTarget = origin(cv::Range(0, bottom_limit), cv::Range(col, col + inputTarget_width));//CaptureRoi(cv::Range(row),cv::Range(col))
							cv::cvtColor(inputTarget, inputTarget, cv::COLOR_BGR2GRAY);
							matchtemplate_error_times = 0;
						}
					}
				}
				if (showpic != 0) {
					//imshow("res", res);
					imshow("frame", frame);
					imshow("frame1", frame1);
					cv::waitKey(1);
				}
			}

			else if ((frame_num % duration) == (duration - 1))
			{
			
			cv::cvtColor(frame, roi_end, cv::COLOR_BGR2GRAY);
			absdiff(roi_start, roi_end, roi_diff);
			cv::Mat roi_diffbin;
			threshold(roi_diff, roi_diffbin, 100, 255, cv::THRESH_BINARY);
			cv::Scalar result = mean(roi_diffbin);
			if (result[0] == 0.0)
			{
				frame_num = 0;

				if (showpic != 0) {
					if (!ones) {
						ones = true;
						std::cout << "畫面沒有移動，不動作" << std::endl;
					}
				}
				//continue;
			}
			}
			frame_num++;
		}
	}
	else//單射源
	{
		while (true)
		{
			//畫基準判斷線供使用者參考 並塗抹X光機下方工具列
			capture >> frame;
			if (frame.empty()) {
				break;
			}
			frame = frame(cv::Range(0, bottom_limit), cv::Range(0, video_col));//CaptureRoi(cv::Range(row),cv::Range(col))

			if (frame_num % duration == 0)
			{
				ones = false;
				cv::cvtColor(frame, roi_start, cv::COLOR_BGR2GRAY);
				cv::Mat origin;
				frame.copyTo(origin);

				if (showpic != 0) {
					line(frame, cv::Point(thr_line, 0), cv::Point(thr_line, bottom_limit), cv::Scalar(255, 0, 0), 4);
				}

				cv::cvtColor(origin, gray, cv::COLOR_BGR2GRAY);
				threshold(gray, bin, 200, 255, cv::THRESH_BINARY_INV);

				if (stoi(LtoR) == 0)//LtoR = 0物件由右至左 <-- 
				{
					if (inputTarget.empty()) //倘使無初始相片可供matchTemplate比對，則自行找
					{
						int col = getAccument_new(thr_line, stoi(LtoR));
						if (showpic != 0) {
							line(frame, cv::Point(col, 0), cv::Point(col, bottom_limit), cv::Scalar(0, 0, 255), 4);
						}
						res = origin(cv::Range(0, bottom_limit), cv::Range(thr_line, col));
						inputTarget = origin(cv::Range(0, bottom_limit), cv::Range(col - inputTarget_width, col));//CaptureRoi(cv::Range(row),cv::Range(col))
						cv::cvtColor(inputTarget, inputTarget, cv::COLOR_BGR2GRAY);
					}
					/// Create the result matrix
					int result_cols = video_col - inputTarget.cols + 1;
					int result_rows = video_row - inputTarget.rows + 1;
					cv::Mat result(result_cols, result_rows, CV_32FC1);

					matchTemplate(gray, inputTarget, result, cv::TM_CCOEFF_NORMED);
					minMaxLoc(result, &minVal, &maxVal, NULL, &Loc);
					//確認是否全為相同之值，以避免導致matchTemplate異常
					if (maxVal == 1) {
						if (showpic != 0) {
							std::cout << "maxVal=1 比對異常" << std::endl;
						}
						maxVal = 0;
					}
					//std::cout << maxVal << endl;
					if (maxVal >= maxVal_thr)//根據比對結果框選邊框，並決定擷取條件。
					{
						matchtemplate_error_times = 0;
						rectangle(frame, Loc, cv::Point(Loc.x + inputTarget.cols, Loc.y + inputTarget.rows), cv::Scalar(0, 255, 0), 4);
						if (video_col - (Loc.x + inputTarget.cols) > min_dist) { //大於一定寬度開始找尋
							cv::Mat A = bin.col(video_col - 10);
							int temp = sum(A)[0] / 255;
							std::cout << "現在累積值 = " << temp << std::endl;
							if (temp < acc_thr) {
								//紀錄時間
								capture_end = clock();
								//截圖時長過短則忽略
								if ((capture_end - capture_start) / CLOCKS_PER_SEC < capture_min_time)
									std::cout << "截圖費時: " << (capture_end - capture_start) / CLOCKS_PER_SEC << ", 時長過短忽略" << std::endl;
								else {
									std::cout << "截圖費時: " << (capture_end - capture_start) / CLOCKS_PER_SEC << std::endl;
									capture_start = capture_end;
									//截圖
									if (showpic != 0) {
										std::cout << "找到最低點截圖" << std::endl;
									}
									res = origin(cv::Range(0, bottom_limit), cv::Range(Loc.x + inputTarget.cols, video_col));

									takePicture_single(res, folder_path);
									image_Name = picfilename_single;
									imagePath = imagePath_single;
									XrayAicheck_process(res);
									
									inputTarget = origin(cv::Range(0, bottom_limit), cv::Range(video_col - inputTarget_width, video_col));
									cv::cvtColor(inputTarget, inputTarget, cv::COLOR_BGR2GRAY);
								}
							}
						}
						if ((Loc.x + inputTarget.cols) <= thr_line) { //如果有過長的寬度則截圖
							//紀錄時間
							capture_end = clock();
							//截圖時長過短則忽略
							if ((capture_end - capture_start) / CLOCKS_PER_SEC < capture_min_time)
								std::cout << "截圖費時: " << (capture_end - capture_start) / CLOCKS_PER_SEC << ", 時長過短忽略" << std::endl;
							else {
								std::cout << "截圖費時: " << (capture_end - capture_start) / CLOCKS_PER_SEC << std::endl;
								capture_start = capture_end;
								//截圖
								if (showpic != 0) {
									std::cout << "未找到最低點 但物件過長 向後尋找可能的物件邊界並截圖" << std::endl;
									//std::cout << "未找到最低點但物件過長截圖" << endl;
								}
								// 若物件過長向後找尋最低點位置
								int col = getAccument_new(Loc.x + inputTarget.cols + min_dist, stoi(LtoR));
								// 20220216 向後找尋最低點
								res = origin(cv::Range(0, bottom_limit), cv::Range(Loc.x + inputTarget.cols, col));
								takePicture_single(res, folder_path);
								image_Name = picfilename_single;
								imagePath = imagePath_single;
								XrayAicheck_process(res);
								inputTarget = origin(cv::Range(0, bottom_limit), cv::Range(col - inputTarget_width, col));
								cv::cvtColor(inputTarget, inputTarget, cv::COLOR_BGR2GRAY);
							}

						}
					}
					else //遺失邊框 找新框
					{
						matchtemplate_error_times += 1;
						if (showpic != 0) {
							std::cout << "遺失邊框" << std::endl;
						}
						if (matchtemplate_error_times == error_template_times) 
						{
							std::cout << "遺失邊框" << matchtemplate_error_times << "次 找新框" << std::endl;
							int col = getAccument_new(thr_line, stoi(LtoR));
							if (showpic != 0) {
								rectangle(frame, cv::Point(col-inputTarget_width, 0), cv::Point(col, bottom_limit), cv::Scalar(0, 0, 255), 4);
							}
							inputTarget = origin(cv::Range(0, bottom_limit), cv::Range(col - inputTarget_width, col));//CaptureRoi(cv::Range(row),cv::Range(col))
							cv::cvtColor(inputTarget, inputTarget, cv::COLOR_BGR2GRAY);
							matchtemplate_error_times = 0;
						}
					}
				}
				// LtoR = 1 物件由左至右 --> 
				else
				{
					if (inputTarget.empty())//倘使無初始相片可供matchTemplate比對，則自行找
					{
						int col = getAccument_new(thr_line, stoi(LtoR));
						if (showpic != 0) {
							line(frame, cv::Point(col, 0), cv::Point(col, bottom_limit), cv::Scalar(0, 0, 255), 4);
						}
						res = origin(cv::Range(0, bottom_limit), cv::Range(col, thr_line));
						inputTarget = origin(cv::Range(0, bottom_limit), cv::Range(col, col + inputTarget_width));//CaptureRoi(cv::Range(row),cv::Range(col))
						cv::cvtColor(inputTarget, inputTarget, cv::COLOR_BGR2GRAY);
					}
					/// Create the result matrix
					int result_cols = video_col - inputTarget.cols + 1;
					int result_rows = video_row - inputTarget.rows + 1;
					cv::Mat result(result_cols, result_rows, CV_32FC1);

					matchTemplate(gray, inputTarget, result, cv::TM_CCOEFF_NORMED);
					minMaxLoc(result, &minVal, &maxVal, NULL, &Loc);

					//確認是否全為相同之值，以避免導致matchTemplate異常
					if (maxVal == 1) {
						if (showpic != 0) {
							std::cout << "maxVal=1 比對異常" << std::endl;
						}
						maxVal = 0;
					}

					if (maxVal >= maxVal_thr)//根據比對結果框選邊框，並決定擷取條件。
					{
						matchtemplate_error_times = 0;
						rectangle(frame, Loc, cv::Point(Loc.x + inputTarget.cols, Loc.y + inputTarget.rows), cv::Scalar(0, 255, 0), 4);
						if (Loc.x > min_dist) { //大於一定寬度開始找尋
							cv::Mat A = bin.col(10);
							int temp = sum(A)[0] / 255;
							//imshow("bin", bin);
							std::cout << "現在累積值 = " << temp << std::endl;
							if (temp < acc_thr)
							{
								//紀錄時間
								capture_end = clock();
								//截圖時長過短則忽略
								if ((capture_end - capture_start) / CLOCKS_PER_SEC < capture_min_time)
									std::cout << "截圖費時: " << (capture_end - capture_start) / CLOCKS_PER_SEC << ", 時長過短忽略" << std::endl;
								else {
									std::cout << "截圖費時: " << (capture_end - capture_start) / CLOCKS_PER_SEC << std::endl;
									capture_start = capture_end;
									//截圖
									if (showpic != 0) {
										std::cout << "找到最低點截圖" << std::endl;
									}
									res = origin(cv::Range(0, bottom_limit), cv::Range(0, Loc.x));

									takePicture_single(res, folder_path);
									image_Name = picfilename_single;
									imagePath = imagePath_single;
									XrayAicheck_process(res);

									inputTarget = origin(cv::Range(0, bottom_limit), cv::Range(0, inputTarget_width));
									cv::cvtColor(inputTarget, inputTarget, cv::COLOR_BGR2GRAY);
								}
							}
						}
						if (Loc.x >= thr_line) { //如果有過長的寬度則截圖
							//紀錄時間
							capture_end = clock();
							//截圖時長過短則忽略
							if ((capture_end - capture_start) / CLOCKS_PER_SEC < capture_min_time)
								std::cout << "截圖費時: " << (capture_end - capture_start) / CLOCKS_PER_SEC << ", 時長過短忽略" << std::endl;
							else {
								std::cout << "截圖費時: " << (capture_end - capture_start) / CLOCKS_PER_SEC << std::endl;
								capture_start = capture_end;
								//截圖
								if (showpic != 0) {
									std::cout << "未找到最低點 但物件過長 向後尋找可能的物件邊界並截圖" << std::endl;
									//std::cout << "未找到最低點但物件過長截圖" << endl;
								}
								// 若物件過長向後找尋最低點位置
								int col = getAccument_new(Loc.x - min_dist, stoi(LtoR));
								// 20220216 向後找尋最低點
								res = origin(cv::Range(0, bottom_limit), cv::Range(col, Loc.x));
								takePicture_single(res, folder_path);
								image_Name = picfilename_single;
								imagePath = imagePath_single;
								XrayAicheck_process(res);
								inputTarget = origin(cv::Range(0, bottom_limit), cv::Range(col, col + inputTarget_width));
								cv::cvtColor(inputTarget, inputTarget, cv::COLOR_BGR2GRAY);
							}
						}
					}
					else //遺失邊框 找新框
					{
						matchtemplate_error_times += 1;
						if (showpic != 0) {
							std::cout << "遺失邊框" << std::endl;
						}
						if (matchtemplate_error_times == error_template_times) 
						{
							std::cout << "遺失邊框" << matchtemplate_error_times << "次 找新框" << std::endl;
							int col = getAccument_new(thr_line, stoi(LtoR));
							if (showpic != 0) {
								rectangle(frame, cv::Point(col, 0), cv::Point(col+inputTarget_width, bottom_limit), cv::Scalar(0, 0, 255), 4);
							}
							inputTarget = origin(cv::Range(0, bottom_limit), cv::Range(col, col + inputTarget_width));//CaptureRoi(cv::Range(row),cv::Range(col))
							cv::cvtColor(inputTarget, inputTarget, cv::COLOR_BGR2GRAY);
							matchtemplate_error_times = 0;
						}
					}
				}
				if (showpic != 0) {
					imshow("frame", frame);
					//imshow("res", res);
					cv::waitKey(1);
				}

			}
			else if ((frame_num % duration) == (duration - 1)) {
				cv::cvtColor(frame, roi_end, cv::COLOR_BGR2GRAY);
				absdiff(roi_start, roi_end, roi_diff);
				threshold(roi_diff, roi_diffbin, 100, 255, cv::THRESH_BINARY);
				cv::Scalar result = mean(roi_diffbin);
				if (result[0] == 0.0) {
					frame_num = 0;
					if (showpic != 0) {
						if (!ones) {
							ones = true;
							std::cout << "畫面沒有移動，不動作" << std::endl;
						}
					}
				}
			}
			frame_num++;
		}
	}
	std::cout << "fin" << std::endl;
	END = clock();
	std::cout << (END - START) / CLOCKS_PER_SEC << std::endl;
	system("pause");
}

