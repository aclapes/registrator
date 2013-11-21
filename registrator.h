#ifndef REGISTRATOR_H
#define REGISTRATOR_H

#include <cstdlib>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <math.h>
#include <dirent.h>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "clipper.hpp"

#define MAP_FROM_RGB 0
#define MAP_FROM_THERMAL 1
#define MAP_FROM_DEPTH 2

class Registrator
{
public: 
	Registrator();

	// Functions for the pixel-to-pixel registration
	float lookUpDepth(cv::Mat depthImg, cv::Point2f dCoord, bool SCALE_TO_THEORETICAL);

	void computeCorrespondingThermalPointFromRgb(std::vector<cv::Point2f> vecRgbCoord, std::vector<cv::Point2f>& vecTCoord, std::vector<cv::Point2f> vecDCoord);

	void computeCorrespondingThermalPointFromRgb(std::vector<cv::Point2f> vecRgbCoord, std::vector<cv::Point2f>& vecTCoord, std::vector<cv::Point2f> vecDCoord, 
			std::vector<double>& depthRectDisplacement, std::vector<int> &bestHom);

	void computeCorrespondingThermalPointFromRgb(std::vector<cv::Point2f> vecRgbCoord, std::vector<cv::Point2f>& vecTCoord, std::vector<cv::Point2f> vecDCoord, 
												std::vector<int> vecDepthInMm, std::vector<double>& depthRectDisplacement, std::vector<double>& minDist, 
												std::vector<int> &bestHom, std::vector<std::vector<int>> &octantIndices, std::vector<std::vector<double>> &octantDistances,
												std::vector<cv::Point3f> &worldCoordPointvector);

	void computeCorrespondingDepthPointFromRgb(std::vector<cv::Point2f> vecRgbCoord,std::vector<cv::Point2f> & vecDCoord);

	void computeCorrespondingRgbPointFromDepth(std::vector<cv::Point2f> vecDCoord,std::vector<cv::Point2f> & vecRgbCoord);

	void computeCorrespondingRgbPointFromThermal(std::vector<cv::Point2f> vecTCoord, std::vector<cv::Point2f>& vecRgbCoord);

	void computeCorrespondingRgbPointFromThermal(std::vector<cv::Point2f> vecTCoord, std::vector<cv::Point2f>& vecRgbCoord, std::vector<double>& minDist, 
												 std::vector<int> &bestHom, std::vector<std::vector<int>> &octantIndices, std::vector<std::vector<double>> &octantDistances);

	void computeCorrespondingRgbPointFromThermal(std::vector<cv::Point2f> vecTCoord, std::vector<cv::Point2f>& vecRgbCoord, std::vector<double>& minDist, 
												 std::vector<int> &bestHom, std::vector<std::vector<int>> &octantIndices, std::vector<std::vector<double>> &octantDistances, 
												 std::vector<cv::Point3f> &worldCoordPointstdvector);

	float backProjectPoint(float point, float focalLength, float principalPoint, float zCoord);

	float forwardProjectPoint(float point, float focalLength, float principalPoint, float zCoord);

	void computeHomographyMapping(std::vector<cv::Point2f>& vecUndistRgbCoord, std::vector<cv::Point2f>& vecUndistTCoord, std::vector<cv::Point2f> vecDCoord,
								std::vector<int> vecDepthInMm, std::vector<double>& minDist, std::vector<int> &bestHom, std::vector<std::vector<int>> &octantIndices,
								std::vector<std::vector<double>> &octantDistances, std::vector<cv::Point3f> &worldCoordPointvector);
	void depthOutlierRemovalLookup(std::vector<cv::Point2f> vecDCoord, std::vector<int> &vecDepthInMm);

	void loadMinCalibrationVars(std::string calFile);

	void drawRegisteredContours(cv::Mat rgbContourImage, cv::Mat& depthContourImage, cv::Mat& thermalContourImage, cv::Mat depthImg, bool preserveColors = false);
	void saveRegisteredContours(cv::Mat depthContourImage, cv::Mat thermalContourImage, std::string depthSavePath, std::string thermalSavePath, std::string imgNbr);
	void buildContourDirectory(std::string rgbLoadPath, std::vector<std::string> &rgbContours);
	void loadRegSaveContours();

	// Functions for showing images and handling mouse operations within OpenCV Highgui
	void markCorrespondingPointsRgb(cv::Point2f rgbCoord);
	void markCorrespondingPointsThermal(cv::Point2f tCoord);
	void markCorrespondingPointsDepth(cv::Point2f dCoord);

	static void updateFrames(int imgNbr, void* obj);

	void markCorrespondingPoints(cv::Point2f rgbCoord, cv::Point2f tCoord, cv::Point2f dCoord, std::vector<double> depthRectDisplacement, int homInd, int MAP_TYPE);
	void showModalityImages(std::string rgbPath,std::string tPath,std::string dPath,int imgNbr);
	void initWindows(int imgNbr);

	void rgbOnMouse( int event, int x, int y, int flags);
	void thermalOnMouse( int event, int x, int y, int flags);
	void depthOnMouse( int event, int x, int y, int flags);

	std::string getRgbImgPath();
	std::string getTImgPath();
	std::string getDImgPath();
	void setRgbImgPath(std::string path);
	void setTImgPath(std::string path);
	void setDImgPath(std::string path);
	void toogleUndistortion(bool undistort);
	void setDiscardedHomographies(std::vector<int> discardedHomographies);


private:

	/*	TrilinearHomographyInterpolator finds the nearest point for each quadrant in 3D space and calculates weights
	based on trilinear interpolation for the input 3D point. The function returns a list of weights of the points
	used for the interpolation */
	void trilinearInterpolator(cv::Point3f inputPoint, std::vector<cv::Point3f> &sourcePoints, std::vector<double> &precomputedDistance, std::vector<double> &weights, 
							   std::vector<int> &nearestSrcPointInd, std::vector<double> &nearestSrcPointDist);

	/* weightedHomographyMapper maps the undistPoint based by a weighted sum of the provided homographies weighted by homWeights */
	void weightedHomographyMapper(std::vector<cv::Point2f> undistPoint, std::vector<cv::Point2f> &estimatedPoint, std::vector<cv::Mat> &homographies, std::vector<double> &homWeights);
	
	void MyDistortPoints(const std::vector<cv::Point2f> src, std::vector<cv::Point2f> & dst, 
						const cv::Mat & cameraMatrix, const cv::Mat &distorsionMatrix);
	void getRegisteredContours(std::vector<cv::Point> contour, std::vector<cv::Point> erodedContour, std::vector<cv::Point>& dContour, std::vector<cv::Point>& tContour, cv::Mat depthImg);


	struct calibrationParams {
		// Depth calibration parameters
		float depthCoeffA; // y = a*x+b
		float depthCoeffB;
	
		int WIDTH;
		int HEIGHT;

		// Camera calibration parameters
		std::vector<int> activeImages;
		cv::Mat rgbCamMat, tCamMat, rgbDistCoeff, tDistCoeff;

		// Rectification matrices and mappings
		std::vector<cv::Mat> planarHom, planarHomInv; 
		std::vector<cv::Point3f> homDepthCentersRgb,homDepthCentersT;
		int defaultDepth;

		// Depth calibration parameters
		cv::Mat rgbToDCalX,rgbToDCalY,dToRgbCalX,dToRgbCalY; 
	} stereoCalibParam;

	struct registrationSettings {
		bool UNDISTORT_IMAGES;
		int nbrClustersForDepthOutlierRemoval;
		int depthProximityThreshold;
		std::vector<int> discardedHomographies;
	} settings;

	cv::Mat rgbImg, tImg, dImg;
	std::string rgbImgPath, tImgPath, dImgPath;
};

#endif // REGISTRATOR_H

// Declare function prototypes
void wrappedRgbOnMouse(int event, int x, int y, int flags, void* ptr);
void wrappedThermalOnMouse(int event, int x, int y, int flags, void* ptr);
void wrappedDepthOnMouse(int event, int x, int y, int flags, void* ptr);
