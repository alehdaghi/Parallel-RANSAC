#ifndef util_h
#define util_h

#include "myRansac/ParallelRansac.h"
#include "egbis/image.h"
#include "egbis/misc.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <boost/timer/timer.hpp>
#include "opencv2/opencv.hpp"
#include "types.h"
#include <iostream>
#include <fstream>
#include <vector>

typedef pcl::PointXYZRGB PointType;
typedef boost::chrono::duration<double> sec;



namespace Utility{

    class Util{
    private:
        double fx , fy ;
        double cx , cy ;

        pcl::PointCloud<PointType> meansCloud;

        Mat tempImg,blurImg;

    public:
        Util(){
            fx = 525.0; fy = 525.0;
            cx = 319.5; cy = 239.5;
        }

        PointType getPos(int u,int v,int d);

        pcl::PointCloud<PointType>::Ptr create_point_cloud_ptr(Mat& depthImage, Mat& rgbImage,PointType& mean);

        cv::Mat create_Normal_image(pcl::PointCloud<pcl::Normal> normal);

        pcl::PointCloud<pcl::Normal>::Ptr segmentPlane(pcl::PointCloud<PointType>::Ptr cloud,
                                                       Mat& normalImg,Mat& segmentPlaneImg,
                                                       std::vector<plane> &planes);

    };
}

#endif

