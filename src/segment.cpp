#include "segment.h"
#include "egbis/image.h"
#include "egbis/misc.h"
#include "egbis/pnmfile.h"
#include "egbis/segment-image.h"


using namespace cv;


image<rgb>* Segmentation::Segment::convertMatToNativeImage(Mat *input){
	int w = input->cols;
	int h = input->rows;
	image<rgb> *im = new image<rgb>(w,h);
	//cout<<w<<" "<<h<<endl;
	for(int i=0; i<input->rows; i++)
	{
		for(int j=0; j<input->cols; j++)
		{
			rgb curr;
			Vec3b intensity = input->at<Vec3b>(i,j);
			curr.b = intensity.val[0];
			curr.g = intensity.val[1];
			curr.r = intensity.val[2];
			//cout<<i<<" "<<j<<endl;
			im->data[i*w+j] = curr;
		}
	}
	return im;
}

Mat Segmentation::Segment::convertNativeToMat(image<rgb>* input){
	int w = input->width();
	int h = input->height();
	//cout<<w<<" "<<h<<endl;
	Mat output(Size(w,h),CV_8UC3);

	for(int i =0; i<w; i++){
		for(int j=0; j<h; j++){
			rgb curr = input->data[i+j*w];
			Vec3b rgb(curr.b,curr.g,curr.r);
			output.at<Vec3b>(j,i) = rgb;
			
		}
	}

	return output;
}

Mat Segmentation::Segment::runEgbisOnMat(Mat *input, float sigma, float k, int min_size, int *numccs,vector<vector<int> > &X,vector<vector<int> > &Y) {
	int w = input->cols;
	int h = input->rows;
	Mat output(Size(w,h),CV_8UC3);
    int len = (int)ceil(20) + 1;
	Mat blur;
    GaussianBlur( *input, blur, Size( len, len ), sigma, 0.0);

    //imshow("img",*input);
    //imshow("cvGuss",blur);

	// 1. Convert to native format
    image<rgb> *nativeImage = convertMatToNativeImage(&blur);
	// 2. Run egbis algoritm
	image<rgb> *segmentetImage = segment_image(nativeImage, sigma, k, min_size, numccs,X,Y);
	// 3. Convert back to Mat format
	output = convertNativeToMat(segmentetImage);

    //imshow("guss",output);
    //waitKey(0);
	return output;
}

Mat Segmentation::Segment::showHistogram( Mat src )
{

	Mat  dst;

	/// Load image
	//src = imread("standford\\rgb\\000001.png",1);

	if( !src.data )
	{ return src; }

	/// Separate the image in 3 places ( B, G and R )
	vector<Mat> bgr_planes;
	split( src, bgr_planes );

	/// Establish the number of bins
	int histSize = 20;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/histSize );

	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	/// Draw for each channel
	for( int i = 1; i < histSize; i++ )
	{
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
			  Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
			  Scalar( 255, 0, 0), 2, 8, 0  );
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
			  Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
			  Scalar( 0, 255, 0), 2, 8, 0  );
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
			  Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
			  Scalar( 0, 0, 255), 2, 8, 0  );
	}

	/// Display
	//namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
	//imshow("calcHist Demo", histImage );

	//waitKey(0);

	return histImage;
}
