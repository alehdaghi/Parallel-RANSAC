#ifndef segment_h
#define segment_h

#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include "egbis/misc.h"
using cv::Mat;
using namespace std;
template <class T> class image;


namespace Segmentation{
    class Segment{
    public:
        Segment(){

        }

        image<rgb>* convertMatToNativeImage(Mat *input);
        Mat convertNativeToMat(image<rgb>* input);
        Mat runEgbisOnMat(Mat *input, float sigma, float k, int min_size,
                          int *numccs,vector<vector<int> > &X,vector<vector<int> > &Y);
        Mat showHistogram( Mat src );

    };
}

#endif
