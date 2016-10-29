#include "util.h"
#include "myintegralimagenormalestimation.h"
#include "segment.h"

using namespace std;


PointType Utility::Util::getPos(int u,int v,int d) {
	PointType p;
	p.z = d / 1000.0;
	p.x = (u - cx) * p.z / fx;
	p.y = (v - cy) * p.z / fy;
	return p;
}

pcl::PointCloud<PointType>::Ptr Utility::Util::create_point_cloud_ptr(Mat& depthImage,
                                                                     Mat& rgbImage, PointType& mean){

	pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
	cloud->width = depthImage.cols; //Dimensions must be initialized to use 2-D indexing
	cloud->height = depthImage.rows;
	cloud->resize(cloud->width*cloud->height);
	int min_depth = INT_MAX;
	int num_of_points_added = 0;
	for(int v=0; v< depthImage.rows; v++){ //2-D indexing
		for(int u=0; u< depthImage.cols; u++) {
            cv::Vec3b bgrPixel = rgbImage.at<cv::Vec3b>(v, u);
			int d = depthImage.at<int16_t>(v,u);
			if(d<0.01 /*|| bgrPixel.dot(bgrPixel)<1*/)
				continue;
			PointType p = getPos(u,v,d);
			if(fabs(p.z)+fabs(p.y)+fabs(p.x)>50)
				//cout<<u<<" "<<v<<d<<endl;
				continue;

			p.b = bgrPixel[0];
			p.g = bgrPixel[1];
			p.r = bgrPixel[2];
			mean.x+=p.x;
			mean.y+=p.y;
			mean.z+=p.z;
			cloud->at(u,v) = p;
			num_of_points_added++;
		}
	}
	mean.x/=num_of_points_added;
	mean.y/=num_of_points_added;
	mean.z/=num_of_points_added;
	
	return cloud;
}

cv::Mat Utility::Util::create_Normal_image(pcl::PointCloud<pcl::Normal> normal){
    cv::Mat res(normal.height,normal.width,CV_64FC3);

	for(int v=0; v< res.cols; v++){ //2-D indexing
		//        cout<<v<<endl;
		for(int u=0; u< res.rows; u++){
			pcl::Normal nor=normal.at(v,u);
            cv::Vec3d bgrPixel((nor.normal_x+1)/2,(nor.normal_y+1)/2,(nor.normal_z+1)/2);
            res.at<cv::Vec3d>(u,v)=bgrPixel;
			//
		}
	}
	return res;
}

pcl::PointCloud<pcl::Normal>::Ptr Utility::Util::segmentPlane(pcl::PointCloud<PointType>::Ptr cloud,
                                                             Mat& normalImg,Mat& segmentPlaneImg,
                                                             std::vector<plane> &planes){

	vector<myPair> contoursIndex;
	//pcl::PointCloud<PointType>::Ptr cloud = create_point_cloud_ptr(cap_depth, cap_rgb);


	//meansCloud=*cloud;
	meansCloud.width = cloud->width;
	meansCloud.height = cloud->height;
	//cout<<"main "<<cloud->width<<endl;
	meansCloud.resize(cloud->width*cloud->height);

	boost::timer::cpu_timer timer1;
	timer1.start();
	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
	MyIntegralImageNormalEstimation<PointType, pcl::Normal> ne;
	ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
	ne.setMaxDepthChangeFactor(0.04f);
	ne.setNormalSmoothingSize(10.0f);
	ne.setInputCloud(cloud);
	PointType B;
	ne.setMeanCloudOutPtr(&meansCloud);
	//cout<<"main "<<&meansCloud<<endl;
	ne.compute(*normals);

	sec seconds = boost::chrono::nanoseconds(timer1.elapsed().user);
	cout << "timeNormal:" <<seconds.count() << std::endl;

	tempImg=create_Normal_image(*normals);
	tempImg.convertTo(normalImg,CV_8UC3,255.0);
	double k=20;
	blurImg=normalImg.clone();
	bilateralFilter(normalImg,blurImg,k,k*2,k/2);

	
  // copies all inliers of the model computed to another PointCloud
	//pcl::copyPointCloud<PointType>(*cloud, inliers, *final);
	

	normalImg=blurImg;

	//imshow("normal255",normalImg);
	
	//imshow("blur",blurImg);
    vector<vector<int> > X;
    vector<vector<int> > Y;
	int num_ccs;
	timer1.stop();
	timer1.start();
    Segmentation::Segment segment;
    segmentPlaneImg = segment.runEgbisOnMat(&blurImg, 0.95, 400, 200, &num_ccs,X,Y);
	//cout<<X.size()<<endl;
	X.resize(num_ccs);
	Y.resize(num_ccs);
	seconds = boost::chrono::nanoseconds(timer1.elapsed().user);
	cout << "timeSegment:" <<seconds.count() << std::endl;
	//imshow("seg",segmentPlaneImg);
	//waitKey(0);
	contoursIndex.resize(num_ccs);
	for(int i=0;i<num_ccs;i++){
		contoursIndex[i].size=X[i].size();
		contoursIndex[i].index=i;
	}
	std::sort(contoursIndex.begin(),contoursIndex.end());
	reverse(contoursIndex.begin(),contoursIndex.end());
	//vector<vector<Point> > contours;
    vector<cv::Vec4i> hierarchy;
	int len=30;
	planes.resize(len);
	int ii=0;
	static plane p;
	
	for(int i=0;ii<len && i<X.size();i++){
		p.Inliers.clear();
		p.points.clear();
		p.hulls.clear();
		p.segs.clear();
		//vector<Point> ps;
		int index=contoursIndex[i].index;
		pcl::Normal nor(0,0,0);
		double fx=0,fy=0,fz=0;
		
        cv::Vec3b bgrPixel = normalImg.at<cv::Vec3b>(cv::Point(X[index][0],Y[index][0]));
		
		int cc=0;
		stringstream ss("seg ");
		ss<<index+1<<endl;
		
		
		//putText(segmentPlaneImg,ss.str(),Point(X[index][100],Y[index][100]), FONT_HERSHEY_COMPLEX_SMALL, 0.5, cvScalar(200,200,250), 1, CV_AA);
		//putText(segmentPlaneImg,ss.str(),Point(X[index][5200],Y[index][3300]), FONT_HERSHEY_COMPLEX_SMALL, 0.5, cvScalar(100,150,200), 1, CV_AA);
        cv::Point pCenter(0,0);
        vector<cv::Point> seg;
		
		
		for(int j=0;j<X[index].size();j++){

			nor=normals->at(X[index][j],Y[index][j]);
            if(__isnan(nor.normal_x) || __isnan(nor.normal_y) || __isnan(nor.normal_z) )
                continue;

			PointType pp=cloud->at(X[index][j],Y[index][j]);
            pCenter+=cv::Point(Y[index][j],X[index][j]);
            seg.push_back(cv::Point(X[index][j],Y[index][j]));
            p.points.push_back(cv::Point3f(pp.x,pp.y,pp.z));
            p.hist.add(cv::Vec3b(pp.r,pp.g,pp.b));
            p.segs.push_back(cv::Point(Y[index][j],X[index][j]));
			cc++;
			fx+=nor.normal_x;
			fy+=nor.normal_y;
			fz+=nor.normal_z;
			
		}
		convexHull(Mat(seg),p.hulls);
		
		if(cc==0)
			continue;
		fx/=(cc);
		fy/=(cc);
		fz/=(cc);
		pCenter.x/=cc;
		pCenter.y/=cc;
		//Vec3b col=segmentPlaneImg.at<Vec3b>(pCenter);
		//cout<<"ii "<<ii<<" "<<Point(X[index][0],Y[index][0])<<endl;
        cv::Vec3b col1=segmentPlaneImg.at<cv::Vec3b>(Y[index][110],X[index][120]);
		p.color=col1;
		//putText(segmentPlaneImg,ss.str(),pCenter, FONT_HERSHEY_COMPLEX_SMALL, 0.9, cvScalar(255,150,200), 1, CV_AA);
		//pCenter=Point(480,640)-pCenter;
		/*if(ii<10){
		rectangle(segmentPlaneImg,pCenter-Point(15,10),pCenter+Point(10,10),cvScalar(255,255,255),CV_FILLED);
		putText(segmentPlaneImg,ss.str(),pCenter, FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(col1[0],col1[1],col1[2]), 1, CV_AA);
		}*/
		p.normal=Point3f(fx,fy,fz); 
		
		planes[ii]=p;
		
		ii++;
	}
	sort(planes.begin(),planes.end());
	reverse(planes.begin(),planes.end());
 
	// Find contours
	
 
	// Find the convex hull object for each contour
	//vector<vector<Point> >hull( contours.size() );
	//for( int i = 0; i < planes.size(); i++ )
	//{  
	//	convexHull( Mat(planes[i].points), planes[i].hulls, false ); 
	//}
 //
	//// Draw contours + hull results
	//RNG rng;
	//drawing = Mat::zeros( cap_depth.size(), CV_8UC3 );
	//for( int i = 0; i< planes.size(); i++ )
	//{
	//	
	//	
	//	//drawContours( drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );

	//	Vec3b bgrPixel = segmentPlaneImg.at<Vec3b>(planes[i].points[10]);
	//	//cout<<bgrPixel<<endl;
	//	Scalar color = Scalar( bgrPixel[0], bgrPixel[1], bgrPixel[2] );
	//	vector<vector<Point> >hull;
	//	hull.push_back(planes[i].hulls);
	//	//drawContours( drawing, hull, 0, color, 1, 8, vector<Vec4i>(), 0, Point() );
	//	vector<vector<Point> >contour;
	//	contour.push_back(planes[i].points);
	//	drawContours( drawing, contour, 0, color, 1, 8, vector<Vec4i>(), 0, Point() );
	//	

	//	
	//}
	//waitKey(0);
 // Show in a window
	
	imshow( "Hull demo", segmentPlaneImg );
	return normals;
	//cout<<"frame1"<<endl;

	//Mat egbisImageColor = runEgbisOnMat(&cap_rgb, 0.95, 400, 200, &num_ccs);
	//imshow("segmentPlane",segmentPlaneImg);

	//imshow("segmentRgb",egbisImageColor);
	//waitKey(0);
	//boost::this_thread::sleep (boost::posix_time::milliseconds (4000));
}


//vector<Matrix> loung_Trajectory(int len=100){
//	ifstream fin("lounge_trajectory.log");
//	Matrix T(4,4);
//	int a;
//	vector<Matrix> v;
//	for(int i=0;i<len;i++){
//		fin>>a>>a>>a;
//		fin>>T;
		
//		v.push_back(T);
//	}
//	return v;
//}

//vector<Matrix> kinfu_Trajectory(int len=40){
//	ifstream fin("kinfu_poses5.txt");
//	Matrix T(4,4);
//	int a;
//	vector<Matrix> v;
//	for(int i=0;i<len;i++){
//		fin>>T;
//		v.push_back(T);
//	}
//	return v;
//}

//vector<Matrix> teddy_Trajectory(int len=40){
//	ifstream fin("D:\\database\\rgbd_dataset_freiburg1_teddy\\groundtruth.txt");
//	float tx,ty,tz,qx,qy,qz,qw;

//	Matrix T(4,4);
//	int a;
//	vector<Matrix> v;
//	for(int i=0;i<len;i++){
//		fin>>tx>>ty>>tz>>qx>>qy>>qz>>qw;
//		v.push_back(T);
//	}
//	return v;
//}
