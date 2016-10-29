#ifndef TYPES_H
#define TYPES_H
#include <vector>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>

using cv::Mat;
using cv::Point3f;
using std::cout;
using std::endl;
struct histogram{
	std::vector<int> R,G,B;
	int len;
	int size;
	histogram(){
		len=32;
		size=0;
		R.resize(len);
		G.resize(len);
		B.resize(len);
		for(int i=0;i<len;i++)
			R[i]=G[i]=B[i]=0;
	}
	histogram(int l){
		len=l;
		size=0;
		R.resize(l);
		G.resize(l);
		B.resize(l);
		for(int i=0;i<l;i++)
			R[i]=G[i]=B[i]=0;
	}
	void add(cv::Vec3b c)
	{
		size++;
		float ff=256.0/len;
		R[int(c[0]/ff)]++;
		G[int(c[1]/ff)]++;
		B[int(c[2]/ff)]++;
	}
	double distance(histogram hist) const{
		double err=0;
		double m=(size+hist.size)/2.0;
		
		if(len!=hist.len)
			return -1;
		for(int i=0;i<len;i++){
			double a=(R[i]-hist.R[i])*(R[i]-hist.R[i])+(G[i]-hist.G[i])*(G[i]-hist.G[i])+(B[i]-hist.B[i])*(B[i]-hist.B[i]);
			err=err+a/(m*m);
		}
		return (err/3.0);
	}
};
struct plane{

	plane(){
		isCenter=false;
		w=1;
	}
	~plane(){
		points.clear();
		Inliers.clear();
		hulls.clear();
		hulls3d.clear();
	}
	cv::Point3f normal;
	float d;
	std::vector<cv::Point3f> points;
	std::vector<cv::Point3f> Inliers;
	std::vector<cv::Point> hulls;
	std::vector<cv::Point3f> hulls3d;
	std::vector<cv::Point> segs;
	double w;
	double area;
	histogram hist;
	cv::Vec3b color;
	int size(){
		return points.size();
	}
    bool operator < (const plane& p) const {
		
		//return w*Inliers.size()<p.w*p.Inliers.size();
		return w*points.size()<p.w*p.points.size();
	}

	cv::Point3f cent;
	bool isCenter;
	cv::Point3f center(){
		if(isCenter)
			return cent;
		isCenter=true;
		for(int i=0;i<Inliers.size();i++){
			cent.x+=Inliers[i].x;
			cent.y+=Inliers[i].y;
			cent.z+=Inliers[i].z;
		}
		cent.x/=Inliers.size();cent.y/=Inliers.size();cent.z/=Inliers.size();
		return cent;

	}
	void makeHull3d(){
		//double fx = 525.0, fy = 525.0; // default focal length
		//double cx = 319.5, cy = 239.5; // default optical center
		//hulls3d.clear();
		//for(int i=0;i<hulls.size();i++){
		//	float A=normal.x*(hulls[i].x-cx)/fx;
		//	float B=normal.y*(hulls[i].y-cy)/fy;
		//	float C=normal.z;
		//	float depth=-d/(A+B+C);
		//	cv::Point3f p((hulls[i].x-cx)/fx*depth,(hulls[i].y-cy)/fy*depth,depth);
		//	//cout<<p<<endl;
		//	hulls3d.push_back(p); 
		//	//cout<<hulls[i]<<endl;
		//	//cout<<A<<" "<<B<<" "<<C<<" "<<depth<<endl;
		//	//cout<<normal.dot(p)+d<<endl;;
		//}
		cv::Point3f newN(0,0,1);
		cv::Mat N1=cv::Mat(normal).reshape(1);
		cv::Mat N2=cv::Mat(newN).reshape(1);
		cv::Mat R(Mat::eye(3,3,CV_32FC1));
		cv::Mat H=N1*N2.t();
		cv::SVD svd(H);

		if(cv::norm(normal.cross(newN))>0.001){
			cv::Mat DIAG = cv::Mat::zeros(3, 3, CV_32FC1);
			cv::Mat C = DIAG.diag(0);
			C.at<float>(0) = 1;	C.at<float>(1) = 1;	C.at<float>(2) = (float)determinant(svd.vt.t()*svd.u.t());
			R=svd.vt.t()*DIAG*svd.u.t();

			//cout<<H<<endl;
			//cout<<R<<endl;
			//cout<<svd.vt.t()<<endl;
			//cout<<svd.u<<endl;
		}
		Mat P=Mat(Inliers).reshape(1).t();
		P=R*P;
		std::vector<cv::Point3f> temp3;
		P=P.t();
		P=P.reshape(3);
		P.copyTo(temp3);

		std::vector<cv::Point2f> temp5;
		for(int i=0;i<temp3.size();i++)
			temp5.push_back(cv::Point2f(temp3[i].x,temp3[i].y));
		
		std::vector<cv::Point2f> temp4;
		cv::convexHull(Mat(temp5),temp4);
		area=cv::contourArea(temp4 ); 
		float Z=temp3[0].z;
		temp3.clear(); 
		for(int i=0;i<temp4.size();i++)
			temp3.push_back(cv::Point3f(temp4[i].x,temp4[i].y,Z));
		P=Mat(temp3).reshape(1).t();
		P=R.t()*P;
		P=P.t();
		P=P.reshape(3);
		P.copyTo(hulls3d);
	}
	float distance(cv::Point3f p){
		return fabs(normal.dot(p)+d);
	}
	float distance(plane p){
		float sum1=0,sum2=0;
		for(int i=0;i<p.hulls3d.size();i++)
			sum1+=distance(p.hulls3d[i]);
		sum1/=p.hulls3d.size();
		for(int i=0;i<hulls3d.size();i++)
			sum2+=p.distance(hulls3d[i]);
		sum2/=hulls3d.size();
		return sum1+sum2;
	}

};

struct greaterPlane
{
    
    bool operator()(plane const &a, plane const &b) const {
		return a.w*a.area>b.w*b.area;
	}
};

struct myPair{
	int size;
	int index; 
    bool operator<(const myPair& p) const
	{
		return size<p.size;
	}
};

#endif
