#ifndef parallel_ransac_h
#define parallel_ransac_h

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <fstream>
#include "typesCuda.h"
#include <boost/timer/timer.hpp>
#include <stdlib.h>
#include "../matrix.h"

#include "..//types.h"


namespace Parallel{
	class RANSAC{
    private:
        segment* segs;
        int num;
    public:
		float ransacCpu(xyz* means,float* percents,segment* segs,int size,int maxIter);		
		float point2Plane(xyz normal,float d,xyz p);
        float point2Plane(cv::Point3f normal,float d,cv::Point3f p);
        double minEigenVector(Utility::Matrix A,cv::Point3f n);
        float percentInliers(cv::Point3f n,float d,plane p);
        float fitBestPlane(plane &p,cv::Point3f& n,float& d);
        int myRansac(std::vector<plane> &planes);



	};

}

#endif
