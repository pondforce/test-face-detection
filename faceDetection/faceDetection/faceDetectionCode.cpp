#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


Mat detectAndDisplay(Mat image);


String face_cascade_name = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml";
//String eyes_cascade_name = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
//CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);


int main(int argc, const char** argv)
{
	Mat frame1, frame2;

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading\n"); return -1; };
	//if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading\n"); return -1; };

	//-- 2. Read the video stream
	VideoCapture capture1, capture2;
	capture1.open(2);
	capture2.open(1);

	while (true)
	{
		capture1 >> frame1;
		capture2 >> frame2;

		//-- 3. Apply the classifier to the frame
		if (!frame1.empty()) //&& !frame2.empty())
		{
			detectAndDisplay(frame1);
			detectAndDisplay(frame2);
			imshow("Camera 1", frame1);
			imshow("Camera 2", frame2);
		}
		else
		{
			printf(" --(!) No captured frame -- Break!"); break;
		}

		int c = waitKey(10);
		if ((char)c == 'q') { break; }
	}

	return 0;
}

Mat detectAndDisplay(Mat image)
{

	Mat frame = image;
	std::vector<Rect> faces;
	//std::vector<Rect> eyes;
	Mat frame_gray;

	int inHeight = 120;
	int inWidth = 0;
	int frameHeight = frame.rows;
	int frameWidth = frame.cols;

	if (!inWidth)
		inWidth = (int)((frameWidth / (float)frameHeight) * inHeight);

	float scaleHeight = frameHeight / (float)inHeight;
	float scaleWidth = frameWidth / (float)inWidth;

	resize(frame, frame_gray, Size(inWidth, inHeight));
	cvtColor(frame_gray, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		float x1 = (float)(faces[i].x * scaleWidth);
		float y1 = (float)(faces[i].y * scaleHeight);
		float x2 = (float)((faces[i].x + faces[i].width) * scaleWidth);
		float y2 = (float)((faces[i].y + faces[i].height) * scaleHeight);
		rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(255, 0, 0), (int)(frameHeight / 150.0), 4);
		cout << "Face Position :: X = " << faces[0].x << " , " << "Y = " << faces[0].y << endl;
		//Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		//ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 0), 4, 8, 0);

		/*
		Mat faceROI = frame_gray(faces[i]);

		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point center((faces[i].x + eyes[j].x + eyes[j].width*0.5)* scaleWidth, (faces[i].y + eyes[j].y + eyes[j].height*0.5)* scaleWidth);
			//Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.1);
			circle(frame, center, radius, Scalar(0, 255, 0), 4, 8, 0);
		}
		*/

	}
	return frame;

}



