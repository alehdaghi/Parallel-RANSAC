
#include "cuda_runtime.h"


#include <curand.h>
#include <curand_kernel.h>

#include "typesCuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

__device__ static __inline__ int _rand(int len)
{
    return rand()%len;
}




__global__ void ransacKernel(xyz* means,float* ds,float* percents,xyz* segs,int* lens,int* start)
{
    int i = threadIdx.x;
    int j = blockIdx.x;
    int k = blockDim.x;


    float thr=1;
    int N=2;
    const int lenI=lens[j];
    //extern __shared__ xyz segs_shared[];

    curandState state;

    //for(int ii=0;ii<N;ii++)
    {
        xyz mean;


        curand_init((unsigned int) clock64()*i*k+j, 0, 0, &state);

        int i1=curand(&state)%lenI;
        int i2=curand(&state)%lenI;
        int i3=curand(&state)%lenI;
        xyz p1=segs[start[j]+i1];
        xyz p2=segs[start[j]+i2];
        xyz p3=segs[start[j]+i3];

        xyz u;u.x=p1.x-p2.x;u.y=p1.y-p2.y;u.z=p1.z-p2.z;
        xyz v;v.x=p1.x-p3.x;v.y=p1.y-p3.y;v.z=p1.z-p3.z;

        float a=u.y*v.z-u.z*v.y;
        float b=u.z*v.x-u.x*v.z;
        float c=u.x*v.y-u.y*v.x;


        //printf("%f %f\n",a*u.x+b*u.y+c*u.z,a*v.x+b*v.y+c*v.z);
        float rsqr=rsqrtf(a*a+b*b+c*c);
        a*=rsqr;b*=rsqr;c*=rsqr;
        /*if(c<-0.001)
        {
            a*=-1*rsqr;b*=-1*rsqr;c*=-1*rsqr;
        }
        else if(c>0.001){
            a*=rsqr;b*=rsqr;c*=rsqr;
        }
        else{
            if(b<-0.001)
            {
                a*=-1*rsqr;b*=-1*rsqr;c*=-1*rsqr;
            }
            else if(b>0.001){
                a*=rsqr;b*=rsqr;c*=rsqr;
            }
            else{
                if(a<-0.001)
                {
                    a*=-1*rsqr;b*=-1*rsqr;c*=-1*rsqr;
                }
                else if(a>0.001){
                    a*=rsqr;b*=rsqr;c*=rsqr;
                }
            }
        }
*/

        float d=-1*(a*p1.x+b*p1.y+c*p1.z);
        if(d<0)
        {
            a*=-1;b*=-1;c*=-1;d*=-1;
        }

        float dis=0;
        int cc=0;
        for(int jj=0;jj<lenI;jj++)
        {
            //mean.x+=segs[start[j]+jj].x;
            //mean.y+=segs[start[j]+jj].y;
            //mean.z+=segs[start[j]+jj].z;
            dis=fabs(a*segs[start[j]+jj].x+b*segs[start[j]+jj].y+c*segs[start[j]+jj].z+d);

            //printf("dis ",dis);
            //if(i==1 && j==1 && dis<thr )
            //	printf("index:%d is n:%f %f %f d:%f p:%f %f %f dis:%f\n",jj,a,b,c,d,segs[start[j]+jj].x,segs[start[j]+jj].y,segs[start[j]+jj].z,dis);
            if(dis<THR_PLANE)
                cc++;

        }
        mean.x=a;
        mean.y=b;
        mean.z=c;
        ds[j*k+i]=d;
        means[j*k+i]=mean;
        percents[j*k+i]=((float)cc)/lenI;

        //printf("b:%d t:%d is %d %d %d\n",j,i,i1,i2,i3);
    }

}




// Helper function for using CUDA to add vectors in parallel.

cudaError_t gpuRansac(xyz* normals,float* ds,float* percents,segment *segs, int size,int maxIter=BLOCK_SIZE){

    size_t allSize=0;
    int numPoints=0;
    int* temp_lens=new int[size];
    int* temp_starts=new int[size];
    xyz* temp_segs=new xyz[size];
    for(int i=0;i<size;i++){
        allSize+=segs[i].len;
        temp_lens[i]=segs[i].len;
    }
    temp_segs=new xyz[allSize];
    int cc=0;
    for(int i=0;i<size;i++){

        temp_starts[i]=cc;
        memcpy(temp_segs+cc,segs[i].points,segs[i].len*sizeof(xyz));
        cc+=segs[i].len;
    }

    xyz* dev_segs;
    xyz* dev_means;
    float* dev_percents;
    float* dev_ds;
    int* dev_lens;
    int* dev_starts;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .

    cudaStatus = cudaMalloc((void**)&dev_segs, allSize*sizeof(xyz));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_lens, size*sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_starts, size*sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_means, size*maxIter*sizeof(xyz));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_ds, size*maxIter*sizeof(float));

    cudaStatus = cudaMalloc((void**)&dev_percents, size*maxIter*sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_segs, temp_segs, allSize*sizeof(xyz), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_lens, temp_lens, size*sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_starts, temp_starts, size*sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    // Launch a kernel on the GPU with one thread for each element.
    ransacKernel<<<size,maxIter,20000>>>(dev_means,dev_ds,dev_percents,dev_segs, dev_lens,dev_starts);






    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds;
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    //printf("%lf",milliseconds);

    xyz* temp_means;
    temp_means=new xyz[size*maxIter];
    float* temp_percents;
    temp_percents=new float[size*maxIter];
    float* temp_ds; temp_ds = new float[size*maxIter];


    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }


    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(temp_means, dev_means, size*maxIter* sizeof(xyz), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(temp_percents, dev_percents, size*maxIter* sizeof(float), cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(temp_ds, dev_ds, size*maxIter* sizeof(float), cudaMemcpyDeviceToHost);


    //xyz* cpu_means=new xyz[size];
    //float* cpu_percents=new float[size*maxIter];

    //boost::timer::cpu_timer timer1;
    //timer1.start();

    //ransacCpu(cpu_means,cpu_percents,temp_segs, temp_lens,temp_starts,size);
    //sec seconds = boost::chrono::nanoseconds(timer1.elapsed().user);
    //printf("timeCpu:%f" ,seconds.count() );
    //ds=new float[size];
    //normals=new xyz[size];
    for(int i=0;i<size;i++){
        float p=0,d=0,d80=0;
        xyz normal;
        normal.x=normal.y=normal.z=0;
        xyz normal80;
        normal80.x=0;normal80.y=0;normal80.z=0;
        for(int j=0;j<maxIter;j++)
        {
            if(temp_percents[i*maxIter+j]>p){
                p=temp_percents[i*maxIter+j];
                d=temp_ds[i*maxIter+j];
                normal=temp_means[i*maxIter+j];
            }
        }
        float sumP=0,w;
        for(int j=0;j<maxIter;j++)
        {
            if(temp_percents[i*maxIter+j]>=p*0.95)
            {
                w=temp_percents[i*maxIter+j];
                normal80.x+=(w*temp_means[i*maxIter+j].x);
                normal80.y+=(w*temp_means[i*maxIter+j].y);
                normal80.z+=(w*temp_means[i*maxIter+j].z);
                d80+=(w*temp_ds[i*maxIter+j]);
                sumP+=w;
            }

        }
        if(fabs(sumP)<0.1)
            sumP=1;
        normal80.x/=sumP;
        normal80.y/=sumP;
        normal80.z/=sumP;
        d80/=sumP;
        ds[i]=d;
        normals[i]=normal;
        percents[i]=p;
        //printf("plane %d percent:%f \n",i,sumP);
    }
        //printf("mean of %d is gpu: %f %f %f cpu: %f %f %f\n",i,
        //	means[i*maxIter+10].x,means[i*maxIter+20].y,means[i*maxIter+6].z,
        //	cpu_means[i].x,cpu_means[i].y,cpu_means[i].z);

Error:
        cudaFree(dev_segs);
        cudaFree(dev_lens);
        cudaFree(dev_means);
        cudaFree(dev_ds);
        cudaFree(dev_percents);
        cudaFree(dev_starts);

        delete temp_segs;
        delete temp_lens;
        delete temp_ds;
        delete temp_means;
        delete temp_starts;

        delete temp_percents;


    return cudaStatus;
}



__global__ void meanKernel(xyz* means,xyz* segs,int* lens,int* start)
{
    int i = threadIdx.x;
    int j = blockIdx.x;
    int k = blockDim.x;


    float thr=0.2;
    int N=10;
    const int lenI=lens[j];
    //extern __shared__ xyz segs_shared[];


    //for(int ii=0;ii<N;ii++)
    {
        xyz mean;
        mean.z=mean.y=mean.x=0;
        for(int jj=0;jj<lenI;jj++)
        {
            mean.x+=segs[start[j]+jj].x;
            mean.y+=segs[start[j]+jj].y;
            mean.z+=segs[start[j]+jj].z;
        }
        mean.x/=lenI;
        mean.y/=lenI;
        mean.z/=lenI;
        means[j*k+i]=mean;
    }

}

