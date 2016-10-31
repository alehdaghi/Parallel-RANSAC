
//#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>

//#include <pcl/features/integral_image_normal.h>


#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <boost/timer/timer.hpp>
#include <omp.h>


#include "util.h"
#include "myRansac/ParallelRansac.h"


using namespace cv;
using namespace pcl;
using namespace std;

typedef pcl::PointXYZRGB PointType;
typedef boost::chrono::duration<double> sec;




int32_t num1 = 15;
class frame{
	public:
	vector<plane> planes;
	pcl::PointCloud<PointType>::Ptr cloud;
	Mat rgb,normal,seg;
	PointType center;

	frame(){
	}
	frame(const frame& f):cloud(new pcl::PointCloud<PointType>()),rgb(f.rgb),normal(f.normal),seg(f.seg),center(f.center){
		*cloud=*f.cloud;
		planes=f.planes;
	}
	void save(string file,string folder){
		//ofstream cout(file+".txt");
        //ofstream cout2(folder+"\\hulls\\"+file+".txt");
		cout<<folder<<"\\hulls"+file+".txt"<<endl;
		//cout<<num1<<endl;
		num1=min((int)planes.size(),15);
        cout<<num1<<endl;
		
		planes.resize(min((int)planes.size(),num1+5));
		sort(planes.begin(),planes.end(),greaterPlane());

        cv::Mat segs(480, 640, CV_8U, Scalar::all(0));
		float sum=0;
		for(int i=0;i<num1;i++)
			sum+=planes[i].area*planes[i].w;
		for(int i=0;i<num1;i++)
		{
			//cout<<planes[i].normal<<" "<<planes[i].d<<endl;
			
            //cout<<planes[i].normal.x<<" "<<planes[i].normal.y<<" "<<planes[i].normal.z<<
                //" "<<planes[i].d<<" "<<(planes[i].area*planes[i].w)/(double)sum<<endl;
			//cout<<planes[i].Inliers.size()<<endl;
			//for(int j=0;j<planes[i].Inliers.size();j++)
				//cout<<planes[i].Inliers[j].x<<" "<<planes[i].Inliers[j].y<<" "<<planes[i].Inliers[j].z<<" ";
			//cout<<endl;
            //cout<<planes[i].hulls3d.size()<<endl;
            //for(int j=0;j<planes[i].hulls3d.size();j++)
                //cout<<planes[i].hulls3d[j].x<<" "<<planes[i].hulls3d[j].y<<" "<<planes[i].hulls3d[j].z<<" ";
            //cout<<endl;
			for(int j=0;j<planes[i].segs.size();j++)
                segs.at<int8_t>(planes[i].segs[j].x, planes[i].segs[j].y) = 255 - i * 10 - 1;
		}
        imwrite(folder+"/"+file+".png",segs);
        imwrite(folder+"/"+file+"n.png",normal);
        imwrite(folder+"/"+file+"s.png",seg);
	}
	~frame(){
		planes.clear();
		
	}
};


void makeFrame(int i,frame& f){
	stringstream ss1;
	ss1<<"D:\\database\\stanford\\copyroom\\depth\\"<<setfill('0') << setw(6)<<i<<".png";
	stringstream ss2;
	ss2<<"D:\\database\\stanford\\copyroom\\color\\"<<setfill('0') << setw(6)<<i<<".png";
    Utility::Util util;
	Mat cap_depth = imread(ss1.str(),CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
	f.rgb = imread(ss2.str(),CV_LOAD_IMAGE_ANYCOLOR);
	//cout<<ss1.str()<<endl;
	//imshow("a",f.rgb);
    f.cloud=util.create_point_cloud_ptr(cap_depth, f.rgb,f.center);
	//showCloud(*f.cloud);
	vector<plane> planes;
	
    pcl::PointCloud<Normal>::Ptr normal1 = util.segmentPlane(f.cloud, f.normal, f.seg, f.planes);

	//showCloud(*f.cloud);
	
	//return f;
}
void makeFrameLounge(int i,frame& f){
	stringstream ss1;
    ss1<<"/media/mahdi/My/datasets/stanford/Lounge/depth/"<<setfill('0') << setw(6)<<i<<".png";
	stringstream ss2;
    ss2<<"/media/mahdi/My/datasets/stanford/Lounge/color/"<<setfill('0') << setw(6)<<i<<".png";
    Utility::Util util;
	Mat cap_depth = imread(ss1.str(),CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
	f.rgb = imread(ss2.str(),CV_LOAD_IMAGE_ANYCOLOR);
	//cout<<ss1.str()<<endl;
	//imshow("a",f.rgb);
    f.cloud = util.create_point_cloud_ptr(cap_depth, f.rgb,f.center);
	//showCloud(*f.cloud);
    //vector<plane> planes;
	
    pcl::PointCloud<Normal>::Ptr normal1 = util.segmentPlane(f.cloud,f.normal,f.seg,f.planes);

	//showCloud(*f.cloud);
	
	//return f;
}

pcl::PointCloud<PointType> makePlanesCloud(frame f){
	pcl::PointCloud<PointType> planesCloud;
	
	static int c=0;
	int rr=rand()%255,gg=rand()%255,bb=rand()%255;
	static string s="abcdefghimnopqrstuvwxyz";
	c++;
	for (int k=0;k<15;k++){
		//int r=rand()%255,g=rand()%255,b=rand()%255;
		//int rr=rand()%255,gg=rand()%255,bb=rand()%255;
		//int rr=rand()%255,gg=rand()%255,bb=rand()%255;
		int rr=rand()%255,gg=rand()%255,bb=rand()%255;
		PointType center;
		int len=f.planes[k].Inliers.size();
		for(int i=0;i<len;i++)
		{
			PointType p(rr,gg,bb);
			p.x=f.planes[k].Inliers[i].x;
			p.y=f.planes[k].Inliers[i].y;
			p.z=f.planes[k].Inliers[i].z;
			
			planesCloud.push_back(p);
		}
		center.x=f.planes[k].center().x;center.y=f.planes[k].center().y;center.z=f.planes[k].center().z;
		stringstream ss1;
		ss1<<c<<k;
        //addText3D("n"+ss1.str(), center, rr/256.0, gg/256.0,b b/256.0);
		//addPlane(f.planes[k].normal.x,f.planes[k].normal.y,f.planes[k].normal.z,f.planes[k].d,
			//center.x,center.y,center.z,ss1.str());
        //addPolygon(f.planes[k].hulls3d,f.planes[k].normal.x,f.planes[k].normal.y,f.planes[k].normal.z,f.planes[k].d,ss1.str());
		
	}
	c++;
	return planesCloud;
}

int main(int argc, char* argv[]) {
	
	boost::timer::cpu_timer timer1;
	srand(time(NULL));

	timer1.start();

    frame f;
    Parallel::RANSAC ransac;
    makeFrameLounge(1,f);
    f.save("test","../out/planes");
#pragma omp parallel private(f,ransac)
    {
        for(int i=0;i<96;i+=4)
        {
            int id = omp_get_thread_num();
            int index = (i/4)*4 + id;



            makeFrameLounge(index+1,f);
            stringstream ss;
            ss<<i;
        #pragma omp critical
            {
            boost::timer::cpu_timer timer2;
            cout<<"id: "<<id<<" i:"<< index <<endl;
            ransac.myRansac(f.planes);
            sec seconds = boost::chrono::nanoseconds(timer2.elapsed().user);
            cout << "time Cuda:" <<seconds.count() << std::endl;
        }

            f.save(ss.str(),"../out/planes");

        }
    }
    //showCloud(*cloud1);
    sec seconds = boost::chrono::nanoseconds(timer1.elapsed().user);
    cout << "time All:" <<seconds.count() << std::endl;
	return 0;

//	sec seconds = boost::chrono::nanoseconds(timer1.elapsed().user);
//	cout << "timeSegment:" <<seconds.count() << std::endl;

//	int ll=num1;

//	Matrix R = Matrix::eye(3),R_=Matrix::eye(3);

//	Matrix t(3,1),t_(3,1);
//	Icn icp(1);
//	//pcl::PointCloud<PointType>::Ptr transformed_cloud=frames[0].cloud;
//	pcl::PointCloud<PointType>::Ptr transformed_cloud;//(&makePlanesCloud(frames[0]));
	
//	t_.zero();
//	vector<vector<int>> idxss(41);
	
//	//idxss[1].resize(10,-1);
	

//	pcl::PointCloud<PointType> Final;//=*frames[0].cloud;
//	pcl::PointCloud<PointType> FinalPlane=makePlanesCloud(frames[0]);
//	int a=0,b=5;
//	//ofstream fout("myTraj1.txt");
//	//ifstream fin("Traj.txt");
//	//Matrix T0=kinfuTraj[0];
//	for(int i=0;i<41;i+=15)
//	{
		
//		frames[i].planes.resize(num1);
//		R=Matrix::eye(3);
//		t.zero();
//		idxss[i].resize(num1,-1);
//		//cout<<(i/21)*20<<" "<<i<<" "<<ll<<endl;
//		//icp.fit(frames[i-10].planes,frames[i].planes,R,t,idxss[i].size(),idxss[i]);
//		//fin>>R>>t;
		 
		
//		//cout<<frames[i-1].center<<endl;
//		//cout<<frames[i].center<<endl;
//		//cout<<~t<<endl;
		
//		Matrix T=loungTraj[i];
//		//Final=*myRotate(Final,R,t);
//		//Final=*myRotate(Final,T);
//		Final+=*myRotate(*frames[i].cloud,T);
//		//FinalPlane=*myRotate(FinalPlane,T);
//		//Final=*transformed_cloud+makePlanesCloud(frames[i]);
//		//Final+=*frames[i].cloud;
//		//FinalPlane+=makePlanesCloud(frames[i]);

//		//fout << R << endl << endl;
//		//fout << t << endl << endl;
		
		
//	}
	
//	//fin.close();
//	float sum=0;
//	seconds = boost::chrono::nanoseconds(timer1.elapsed().user);
//	cout << "time:" <<seconds.count() << std::endl;
	
//	Matrix tt(3,1);
//	//tt.val[0][0]=8;
//	//tt.val[1][0]=5;
//	//tt.val[2][0]=10;
//	showCloud(/*FinalPlane+*/*myRotate(Final,Matrix::eye(3),tt));
//	//showCloud(pcl::PointCloud<PointType>());
//	waitKey(0);
	
//	exit(0);
	
}


double delta = 5.0;

// Estimated overlap (see the paper).
double overlap = 0.2;

// Threshold of the computed overlap for termination. 1.0 means don't terminate
// before the end.
double thr = 1.0;

// Maximum norm of RGB values between corresponded points. 1e9 means don't use.
double max_color = 150;

// Number of sampled points in both files. The 4PCS allows a very aggressive
// sampling.
int n_points = 200;

// Maximum angle (degrees) between corresponded normals.
double norm_diff = 90.0;

// Maximum allowed computation time.
int max_time_seconds = 100;

