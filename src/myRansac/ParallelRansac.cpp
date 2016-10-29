#include "ParallelRansac.h"

typedef boost::chrono::duration<double> sec;


extern cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);
extern cudaError_t gpuRansac(xyz* normals,float* ds,float* percents,segment *segs, int size,int maxIter=BLOCK_SIZE);



float Parallel::RANSAC::point2Plane(xyz normal,float d,xyz p)
{
	return fabs(normal.x*p.x+normal.y*p.y+normal.z*p.z+d);
}
float Parallel::RANSAC::point2Plane(cv::Point3f normal,float d,cv::Point3f p)
{
	return fabs(normal.dot(p)+d);
}

double Parallel::RANSAC::minEigenVector(Utility::Matrix A,cv::Point3f n){
    Utility::Matrix u,s,v;
    Utility::Matrix H=(~A)*A*(1.0/A.m);
	//std::cout<<"H: "<<H<<std::endl;
	
	H.svd(u,s,v);
	//std::cout<<u<<std::endl;
	
	n=cv::Point3f(u.val[0][2],u.val[1][2],u.val[2][2]);
	if(n.z<0)
		n*=-1.0;
	return (s.val[0][0]+s.val[0][1])/(s.val[0][0]+s.val[0][1]+s.val[0][2]);
}

float Parallel::RANSAC::percentInliers(cv::Point3f n,float d,plane p){
	float cc=0;
	for(int i=0;i<p.points.size();i++){
		if(point2Plane(n,d,p.points[i])<THR_PLANE/100.0)
			cc+=1.0;
	}
	//std::cout<<n<<" "<<d<<std::endl;
	return cc/p.points.size();
}

float Parallel::RANSAC::fitBestPlane(plane &p,cv::Point3f& n,float& d){

	cv::Point3f c;
    Utility::Matrix A(p.points.size(),3);
		
			
		
	for(int i=0;i<p.Inliers.size();i++)
		c+=p.Inliers[i];

	c=c*(1.0/p.Inliers.size());
	for(int i=0;i<p.Inliers.size();i++){
		A.val[i][0]=p.Inliers[i].x-c.x;
		A.val[i][1]=p.Inliers[i].y-c.y;
		A.val[i][2]=p.Inliers[i].z-c.z;
	}
	
	double w=minEigenVector(A,n);
	std::vector<cv::Point3f> inliers;
	d=-1*n.dot(c);
	if(d<0){
		n*=-1;
		d*=-1;
	}

	p.w=w;
	return percentInliers(n,d,p);

}
int Parallel::RANSAC::myRansac(std::vector<plane> &planes)
{


	num=planes.size();
	segs=new segment[num];
	for(int i=0;i<num;i++){
		segs[i].len=planes[i].points.size();
		segs[i].points=new xyz[segs[i].len];
		//cout<<segs[i].len<<endl;
		for (int j=0;j<segs[i].len;j++){
			//fin>>segs[i].points[j].x>>segs[i].points[j].y>>segs[i].points[j].z;
			
			segs[i].points[j].x=100*planes[i].points[j].x;
			segs[i].points[j].y=100*planes[i].points[j].y;
			segs[i].points[j].z=100*planes[i].points[j].z;
		}
	}

	xyz* normals=new xyz[num];
	float* ds=new float[num];
	float* percents=new float[num];
	cudaError_t cudaStatus;
	std::vector<std::pair<int,int> > list;
	boost::timer::cpu_timer timer1;

    cudaStatus = gpuRansac(normals, ds, percents, segs,num);
    float p=0;

    int a=0;
    for(int i=0;i<num;i++){
        p+=(percents[i]*segs[i].len);
        a+=segs[i].len;
    }
    p/=a;
    timer1.start();

	//printf("\n");
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	
	for(int i=0;i<num;i++)
	{
		//printf("%f %f %f %f size=%d per=%f;\n",
			//normals[i].x,normals[i].y,normals[i].z, ds[i],planes[i].points.size());
		planes[i].d=ds[i]/100;
		planes[i].normal.x=normals[i].x;
		planes[i].normal.y=normals[i].y;
		planes[i].normal.z=normals[i].z;
		cv::Point3f c(0,0,0);

		for(int j=0;j<segs[i].len;j++){
            if(point2Plane(normals[i], ds[i], segs[i].points[j]) < THR_PLANE)
			{
                cv::Point3f p(segs[i].points[j].x/100, segs[i].points[j].y/100, segs[i].points[j].z/100);
				planes[i].Inliers.push_back(p);
                c = c + cv::Point3f(segs[i].points[j].x/100,segs[i].points[j].y/100,segs[i].points[j].z/100);
			}
			
		}
		
		if(planes[i].Inliers.size()<=0)
			continue;
		//std::cout<<planes[i].normal<<" "<<planes[i].d<<std::endl;
		//std::cout<<"len: "<<planes[i].Inliers.size()<<std::endl;
		//std::cout<<"ransac: "<<planes[i].normal<<" "<<planes[i].d<<std::endl;
		planes[i].isCenter=true;
		c=c*(1.0/planes[i].Inliers.size());
		planes[i].cent=c;
		cv::Point3f n;
		float d;
		boost::timer::cpu_timer timer1;
		timer1.start();
		float pF=fitBestPlane(planes[i],planes[i].normal,planes[i].d);
		float pR=percentInliers(planes[i].normal,planes[i].d,planes[i]);
		timer1.stop();
		sec seconds = boost::chrono::nanoseconds(timer1.elapsed().user);
		planes[i].makeHull3d();

	}
    //printf("\n\n");
    delete[] segs;


	return 0;
}
