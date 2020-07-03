#include "custom_flags.hpp"
#include "bodypartextractor.hpp"
#include <math.h>
#include <cmath>
#include <iostream>
#include <dirent.h>
#include <sys/types.h>


// Configuring OpenPose
void BodyPartExtractor::configureWrapper() {

	try {

		// logging_level
		op::checkBool(
			0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
			__LINE__, __FUNCTION__, __FILE__);
		op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
		op::Profiler::setDefaultX(FLAGS_profile_speed);

		// Applying user defined configuration - GFlags to program variables
		// producerType
		op::ProducerType producerType;
		op::String producerString;
		std::tie(producerType, producerString) = op::flagsToProducer(
			op::String(FLAGS_image_dir), op::String(FLAGS_video), op::String(FLAGS_ip_camera), FLAGS_camera,
			FLAGS_flir_camera, FLAGS_flir_camera_index);
		// cameraSize
		const auto cameraSize = op::flagsToPoint(op::String(FLAGS_camera_resolution), "-1x-1");
		// outputSize
		const auto outputSize = op::flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");
		// netInputSize
		const auto netInputSize = op::flagsToPoint(op::String(FLAGS_net_resolution), "-1x368");
		// faceNetInputSize
		const auto faceNetInputSize = op::flagsToPoint(op::String(FLAGS_face_net_resolution), "368x368 (multiples of 16)");
		// handNetInputSize
		const auto handNetInputSize = op::flagsToPoint(op::String(FLAGS_hand_net_resolution), "368x368 (multiples of 16)");
		// poseMode
		const auto poseMode = op::flagsToPoseMode(FLAGS_body);
		// poseModel
		const auto poseModel = op::flagsToPoseModel(op::String(FLAGS_model_pose));
		// JSON saving
		if (!FLAGS_write_keypoint.empty())
			op::opLog(
				"Flag `write_keypoint` is deprecated and will eventually be removed. Please, use `write_json`"
				" instead.", op::Priority::Max);
		// keypointScaleMode
		const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
		// heatmaps to add
		const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
			FLAGS_heatmaps_add_PAFs);
		const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
		// >1 camera view?
		const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1 || FLAGS_flir_camera);
		// Face and hand detectors
		const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
		const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
		// Enabling Google Logging
		const bool enableGoogleLogging = true;

		// Pose configuration (use WrapperStructPose{} for default and recommended configuration)
		const op::WrapperStructPose wrapperStructPose{
			poseMode, netInputSize, outputSize, keypointScaleMode, FLAGS_num_gpu, FLAGS_num_gpu_start,
			FLAGS_scale_number, (float)FLAGS_scale_gap, op::flagsToRenderMode(FLAGS_render_pose, multipleView),
			poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
			FLAGS_part_to_show, op::String(FLAGS_model_folder), heatMapTypes, heatMapScaleMode, FLAGS_part_candidates,
			(float)FLAGS_render_threshold, FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max,
			op::String(FLAGS_prototxt_path), op::String(FLAGS_caffemodel_path),
			(float)FLAGS_upsampling_ratio, enableGoogleLogging };

		this->opWrapper.configure(wrapperStructPose);
		// Face configuration (use op::WrapperStructFace{} to disable it)
		// const op::WrapperStructFace wrapperStructFace{
		//     FLAGS_face, faceDetector, faceNetInputSize,
		//     op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
		//     (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold};
		const op::WrapperStructFace wrapperStructFace{};
		this->opWrapper.configure(wrapperStructFace);
		// Hand configuration (use op::WrapperStructHand{} to disable it)
		// const op::WrapperStructHand wrapperStructHand{
		//     FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
		//     op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
		//     (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold};
		const op::WrapperStructHand wrapperStructHand{};
		this->opWrapper.configure(wrapperStructHand);
		// Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
		// const op::WrapperStructExtra wrapperStructExtra{
		//     FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads};
		const op::WrapperStructExtra wrapperStructExtra{};
		this->opWrapper.configure(wrapperStructExtra);
		// Producer (use default to disable any input)
		// const op::WrapperStructInput wrapperStructInput{
		//     producerType, producerString, FLAGS_frame_first, FLAGS_frame_step, FLAGS_frame_last,
		//     FLAGS_process_real_time, FLAGS_frame_flip, FLAGS_frame_rotate, FLAGS_frames_repeat,
		//     cameraSize, op::String(FLAGS_camera_parameter_path), FLAGS_frame_undistort, FLAGS_3d_views};
		// this->opWrapper.configure(wrapperStructInput);
		// Output (comment or use default argument to disable any output)
		// const op::WrapperStructOutput wrapperStructOutput{
		//     FLAGS_cli_verbose, op::String(FLAGS_write_keypoint), op::stringToDataFormat(FLAGS_write_keypoint_format),
		//     op::String(FLAGS_write_json), op::String(FLAGS_write_coco_json), FLAGS_write_coco_json_variants,
		//     FLAGS_write_coco_json_variant, op::String(FLAGS_write_images), op::String(FLAGS_write_images_format),
		//     op::String(FLAGS_write_video), FLAGS_write_video_fps, FLAGS_write_video_with_audio,
		//     op::String(FLAGS_write_heatmaps), op::String(FLAGS_write_heatmaps_format), op::String(FLAGS_write_video_3d),
		//     op::String(FLAGS_write_video_adam), op::String(FLAGS_write_bvh), op::String(FLAGS_udp_host),
		//     op::String(FLAGS_udp_port)};
		// this->opWrapper.configure(wrapperStructOutput);
		// GUI (comment or use default argument to disable any visual output)
		// const op::WrapperStructGui wrapperStructGui{
		//     op::flagsToDisplayMode(FLAGS_display, FLAGS_3d), !FLAGS_no_gui_verbose, FLAGS_fullscreen};
		// this->opWrapper.configure(wrapperStructGui);
		// Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
		if (FLAGS_disable_multi_thread)
			this->opWrapper.disableMultiThreading();
	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	}
}


// constructor
BodyPartExtractor::BodyPartExtractor() {

	// Configure OpenPose
	op::opLog("Configuring OpenPose...", op::Priority::High);
	this->configureWrapper();

	// Starting OpenPose
	op::opLog("Starting thread(s)...", op::Priority::High);
	this->opWrapper.start();

}


// process input data
void BodyPartExtractor::processImage(cv::Mat inputImage) {
	this->inputImage = inputImage;
	const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(inputImage);
	this->ProcessedData = this->opWrapper.emplaceAndPop(imageToProcess);
}


// print keypoints in the terminal
void BodyPartExtractor::printKeypoints() {

	if (this->ProcessedData != nullptr && !this->ProcessedData->empty()) {
		// Alternative 1
		//op::opLog("Body keypoints: " + this->ProcessedData->at(0)->poseKeypoints.toString(), op::Priority::High);

		// // Alternative 2
		// op::opLog(datumsPtr->at(0)->poseKeypoints, op::Priority::High);

		 // Alternative 3
		 //std::cout << this->ProcessedData->at(0)->poseKeypoints << std::endl;

		  //Alternative 4 - Accesing each element of the keypoints
		 op::opLog("\nKeypoints:", op::Priority::High);
		 const auto& poseKeypoints = this->ProcessedData->at(0)->poseKeypoints;
		 op::opLog("Person pose keypoints:", op::Priority::High);
		 for (auto person = 0 ; person < poseKeypoints.getSize(0) ; person++) {
			 std::cout << " " << std::endl;
		     op::opLog("Person " + std::to_string(person) + " (x, y, score):", op::Priority::High);
			 for (auto bodyPart = 0 ; bodyPart < poseKeypoints.getSize(1) ; bodyPart++) {
			     std::string valueToPrint;
				 for (auto xyscore = 0; xyscore < poseKeypoints.getSize(2); xyscore++) {
				     valueToPrint += std::to_string(   poseKeypoints[{person, bodyPart, xyscore}]   ) + " ";
				 }
				 op::opLog(valueToPrint, op::Priority::High);
			 }
		 }
		 op::opLog(" ", op::Priority::High);
	} else {
		op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
	}
}


// display image
void BodyPartExtractor::displayImage(){
	if (this->ProcessedData != nullptr && !this->ProcessedData->empty()) {
		cv::Mat cvMat = OP_OP2CVCONSTMAT(this->ProcessedData->at(0)->cvOutputData);
		cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", cvMat);
	} else {
		op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
	}
}


// general function to extract a body part defined by two points
// pointOne: anchor point = center of rotation
void BodyPartExtractor::extractBodyPart(int person_idx, int pointOne, int pointTwo, float width_scalar, std::string ID, Skeleton &person) {

	const auto& poseKeypoints = this->ProcessedData->at(0)->poseKeypoints;

	// check if head
	bool head = false;
	if ((pointOne == 1 && pointTwo == 0) || (pointOne == 0 && pointTwo == 1)) {
		head = true;
	}

	// define anchor and extreme points
	std::vector<float> anchor = { poseKeypoints[{person_idx, pointOne, 0}], poseKeypoints[{person_idx, pointOne, 1}], poseKeypoints[{person_idx, pointOne, 2}] }; // {x, y, score}
	std::vector<float> extreme = { poseKeypoints[{person_idx, pointTwo, 0}], poseKeypoints[{person_idx, pointTwo, 1}], poseKeypoints[{person_idx, pointTwo, 2}] }; // {x, y, score}
	if (anchor[2] == 0.f || extreme[2] == 0.f) {
		return;
	}

	// make sure all coordinate are in the image space
	if (anchor[0] < 0 || anchor[0] >= this->inputImage.cols ||
		anchor[1] < 0 || anchor[1] >= this->inputImage.rows ||
		extreme[0] < 0 || extreme[0] >= this->inputImage.cols ||
		extreme[1] < 0 || extreme[1] >= this->inputImage.rows) {
		return;
	}

	// get body part statistics
	float length = sqrt(pow((anchor[0] - extreme[0]), 2) + pow((anchor[1] - extreme[1]), 2));
	double alpha = (atan2(extreme[0] - anchor[0], extreme[1] - anchor[1])); // remember that we are using image coordinates
	float width = length * width_scalar;

	// rotate original image
	cv::Mat rotatedImage;
	if (!head) {
		cv::Point2f center(anchor[0], anchor[1]);
		cv::Mat rot = cv::getRotationMatrix2D(center, -alpha * RAD2DEG, 1.0);
		cv::warpAffine(this->inputImage, rotatedImage, rot, this->inputImage.size());
	} else {
		rotatedImage = this->inputImage;
	}

	// get cropping coordinates knowing body part statistics and center of rotation
	int startX, startY, endX, endY;

	if (!head) {
		startX = MAXIMUM((int)(anchor[0] - width / 2.f), 0);
		startY = MAXIMUM((int)anchor[1], 0);
		endX = MINIMUM((int)(anchor[0] + width / 2.f), this->inputImage.cols - 1);
		endY = MINIMUM((int)anchor[1] + length, this->inputImage.rows - 1);
	} else {
		startX = MAXIMUM((int)(MINIMUM(anchor[0], extreme[0]) - width / 2.f), 0);
		startY = MAXIMUM((int)anchor[1] - length, 0);
		endX = MINIMUM((int)(MAXIMUM(anchor[0], extreme[0]) + width / 2.f), this->inputImage.cols - 1);
		endY = MINIMUM((int)anchor[1] + length, this->inputImage.rows - 1);
	}
	if (startX == endX && startY == endY) {
		return;
	}

	// crop rotated image
	cv::Mat outputImage = rotatedImage(cv::Rect(startX, startY, endX - startX, endY - startY));
	person.body_parts.push_back(outputImage);

	// body part confidence
	float confidence = anchor[2] * extreme[2];
	person.conf_body_parts.push_back(confidence);

	// body part ID
	person.ID_body_parts.push_back(ID);
	return;
}


// we use keypoint score to check it's validity
void BodyPartExtractor::analyzeImage() {

	// image keypoints
	const auto& poseKeypoints = this->ProcessedData->at(0)->poseKeypoints;
	
	// loop through all the people in the image
	this->people.clear();
	for (int p = 0; p < poseKeypoints.getSize(0); p++) {

		// define new person
		Skeleton person;

		// extract body parts
		for (unsigned int i = 0; i < this->partsToExtract.size(); i++) {
			this->extractBodyPart(p, this->partsAnchor[i], this->partsExtreme[i], this->partsWidth[i], this->partsToExtract[i], person);
		}

		// store person object in people
		this->people.push_back(person);
	}
}


// display previously-extracted body parts
void BodyPartExtractor::displayBodyParts() {

	// iterate through each person in the image
	for (unsigned int i = 0; i < this->people.size(); i++) {
		std::cout << "Person " << i << std::endl;
		for (unsigned int b = 0; b < this->people[i].body_parts.size(); b++) {
			cv::imshow(this->people[i].ID_body_parts[b], this->people[i].body_parts[b]);

			std::string confidence;
			confidence += "Confidence " + this->people[i].ID_body_parts[b] + ": " + std::to_string(this->people[i].conf_body_parts[b]);
			std::cout << confidence << std::endl;
		}
		std::cout << std::endl;
		cv::waitKey(0);
	}
}


// store previously-extracted body parts
void BodyPartExtractor::storeBodyParts(std::string &folder, std::string &filename, FILE *dataset) {
	boost::filesystem::path p(filename);
	std::string imageName = p.filename().stem().string();

	// iterate through each person in the image
	for (unsigned int i = 0; i < this->people.size(); i++) {
		for (unsigned int b = 0; b < this->people[i].body_parts.size(); b++) {
			std::string outputFile = folder + this->people[i].ID_body_parts[b] + "/" + imageName + "_p_" + std::to_string(i) + "_c_" + std::to_string(this->people[i].conf_body_parts[b]) + ".jpg";
			cv::imwrite(outputFile, this->people[i].body_parts[b]);
		}
	}

	// identify which body parts could be extracted for each person
	for (unsigned int i = 0; i < this->people.size(); i++) {
		std::string personRow = p.filename().stem().string() + "_p_" + std::to_string(i);
		fprintf(dataset, "%s, ", personRow.c_str());
		for (unsigned int d = 0; d < this->partsToExtract.size(); d++) {
			bool found = false; 
			for (unsigned int b = 0; b < this->people[i].body_parts.size(); b++) {
				if (this->partsToExtract[d] == this->people[i].ID_body_parts[b]) {
					found = true;
					break;
				}
			}
			if (found) {
				fprintf(dataset, "1, ");
			} else {
				fprintf(dataset, "0, ");
			}
		}
		fprintf(dataset, "\n");
	}
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////


// read single file from directory
void FileSystem::readSingle(std::string &file, std::vector<std::string> &files) {
	files.push_back(file);
}


// read multiple files from directory
void FileSystem::readFromDir(std::string &path, std::string &format, std::vector<std::string> &files) {
	boost::filesystem::path p(path);
	boost::filesystem::directory_iterator end_itr;

	// cycle through the directory
	for (boost::filesystem::directory_iterator itr(p); itr != end_itr; ++itr) {

		// If it's not a directory, list it. If you want to list directories too, just remove this check.
		if (is_regular_file(itr->path())) {

			// Check that the file has extension
			if (itr->path().has_extension()) {

				// Extract extension
				std::string extension = itr->path().extension().string();

				// Check we are getting the right extension
				if (format.compare(extension) == 0) {
					files.push_back(itr->path().string());
				}
			}

		} else {

			// if there are subdirectories with images, go through them as well!
			std::string subdir = itr->path().string();
			this->readFromDir(subdir, format, files);
		}
	}
}


// read multiple files from directory
void FileSystem::createDirs(std::string &path, std::vector<std::string> subfolders) {
	this->output_path = path;

	// create directory if needed
	boost::filesystem::path p(path);
	boost::filesystem::create_directory(p);

	// initialize txt with header
	std::string txtFile = path + "datasetInfo.csv";
	this->dataset_f = fopen(txtFile.c_str(), "w+");
	fprintf(this->dataset_f, "filename, ");
	for (unsigned int i = 0; i < subfolders.size(); i++) {
		fprintf(this->dataset_f, "%s, ", subfolders[i]);
	}
	fprintf(this->dataset_f, "\n");

	// create subfolders
	for (unsigned int i = 0; i < subfolders.size(); i++) {
		std::string subpath = path + subfolders[i];
		boost::filesystem::path sp(subpath);
		boost::filesystem::create_directory(sp);
	}
}


// class destructor
FileSystem::~FileSystem() {
	fclose(this->dataset_f);
}


int main(int argc, char *argv[]) {

	// class initialization
	FileSystem fs;
	BodyPartExtractor clase;

	// initialize writing directories
#ifdef STORE_BODYPARTS
	std::string folderOutput = "examples/output_test/"; // relative to the main OpenPose folder
	fs.createDirs(folderOutput, clase.partsToExtract);
#endif

	// read files to process
#if defined(READ_SINGLE_IMG)

	// Option 1: Process single image
	std::string filename = "examples/media/COCO_val2014_000000000241.jpg";
	fs.readSingle(filename, clase.filesToProcess);

#elif defined(READ_FROM_DIR)

	// Option 2: Process multiple images
	std::string format = ".jpeg";
	//std::string path = "examples/media/";
	std::string path = "examples/underground_reid/probe/";
	fs.readFromDir(path, format, clase.filesToProcess);

#elif defined (READ_VIDEO)

	// Option 3: Process video frame by frame
	std::string filename = "examples/media/video.avi";
	boost::filesystem::path path_videos(filename);
	cv::VideoCapture cap(filename);
	if (!cap.isOpened()) {
		std::cout << "Error: Couldn't open video file." << std::endl;
		return 0;
	}

#else
	std::cout << "Error: No file to process." << std::endl;
	return 0;
#endif

	//loop for processing each image stored in a vector
	int i = 0;
	while (true) {

#if defined(READ_SINGLE_IMG) || defined(READ_FROM_DIR)

		// load image from file
		cv::Mat cvImageToProcess = cv::imread(clase.filesToProcess[i]);	

#elif defined (READ_VIDEO)

		// load image from video
		cv::Mat cvImageToProcess;
		cap >> cvImageToProcess;
		if (cvImageToProcess.empty()) {
			break;
		}

		std::string framename = path_videos.filename().stem().string() + "_" + std::to_string(i) + ".jpg";
		clase.filesToProcess.push_back(framename);

#endif

		// process image
		std::cout << "Processing file " << clase.filesToProcess[i] << std::endl;
		clase.processImage(cvImageToProcess);

		// print keyponts
#ifdef PRINT_KEYPOINTS
		clase.printKeypoints();
#endif

		// display keypoint
#ifdef DISPLAY_IMAGES
		clase.displayImage();
#endif

		// extracts desired body parts for each person in the image
		clase.analyzeImage();

		// display body parts
#ifdef DISPLAY_IMAGES
		clase.displayBodyParts();
#endif

		// store body parts
#ifdef STORE_BODYPARTS
		clase.storeBodyParts(folderOutput, clase.filesToProcess[i], fs.dataset_f);
#endif

		// update loop
		i++;
#ifndef READ_VIDEO
		if (i == clase.filesToProcess.size()) {
			break;
		}
#endif
	}

	return 0;
}

