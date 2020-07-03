#ifndef INCLUDE_BODYPARTEXTRACTOR_H_
#define INCLUDE_BODYPARTEXTRACTOR_H_

#include <opencv2/opencv.hpp>
#include <openpose/headers.hpp>
#include <boost/filesystem.hpp>


#define READ_FROM_DIR
//#define READ_SINGLE_IMG
//#define READ_VIDEO
//#define PRINT_KEYPOINTS
#define STORE_BODYPARTS
//#define DISPLAY_IMAGES

#define PI 3.14159265358979
#define RAD2DEG (180.0 / PI)
#define DEG2RAD (PI / 180.0)
#define MAXIMUM(a, b) ((a) > (b) ? (a) : (b))
#define MINIMUM(a, b) ((a) < (b) ? (a) : (b))
#define OFFSET_LEG 0.3 // percentage


class FileSystem {
public:

	FileSystem() {} // inline (empty) constructor
	~FileSystem();
	std::string output_path;
	FILE *dataset_f;

	void readSingle(std::string &file, std::vector<std::string> &files);
	void readFromDir(std::string &path, std::string &format, std::vector<std::string> &files);
	void createDirs(std::string &path, std::vector<std::string> subfolders);
};


class Skeleton {

public:

	Skeleton() {} // inline (empty) constructor

	std::vector<cv::Mat> body_parts;
	std::vector<float> conf_body_parts;
	std::vector<std::string> ID_body_parts;
};


class BodyPartExtractor {

public:

	BodyPartExtractor();

	op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
	std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> ProcessedData;
	cv::Mat inputImage;
	std::vector<Skeleton> people;
	std::vector<std::string> filesToProcess;

	std::vector<std::string> partsToExtract = {"left_arm", "right_arm",
											   "left_thigh", "right_thigh",
											   "left_shin", "right_shin",
											   "left_forearm", "right_forearm", "head"};
	std::vector<int> partsAnchor = { 5, 2, 11, 8, 12, 9, 6, 3, 0 };
	std::vector<int> partsExtreme = { 6, 3, 12, 9, 13, 10, 7, 4, 1 };
	std::vector<float> partsWidth = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 1 };
	
	void configureWrapper();
	void processImage(cv::Mat inputImage);
	void printKeypoints();
	void displayImage();
	void analyzeImage();
	void extractBodyPart(int person_idx, int pointOne, int pointTwo, float width_scalar, std::string ID, Skeleton &person);
	void displayBodyParts();
	void storeBodyParts(std::string &folder, std::string &filename, FILE *dataset);

};

#endif /* INCLUDE_BODYPARTEXTRACTOR_H_ */
