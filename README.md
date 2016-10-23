# Parallel-RANSAC
Extracting planes from RGB-D images in parallel

One of the most significant capabilities of a mobile robot is to understand the environment at least at some level to know how to behave and reach the goal. RGBD images, popularized during past few years, provide rich information about the appearance and the geometry of the environment at the same time; hence, they are quite useful for photo realistic scene reconstruction. However, most of scene reconstruction approaches are offline algorithms and very far from realtime as many use a variant of Iterative Closest Point (ICP) algorithm at their core to match correspondences. Using more abstract geometrical features such as planes enhance the precision of this process but needs a fast plane extraction technique to catch up with frame rate of the camera. In this paper, we propose a parallel extension to famous Random Sample Consensus (RANSAC) algorithm which uses GPU cores to speedup extraction of planes out of point clouds obtained from a Kinect sensor. A graph cut segmentation of the surface normal vectors is used to guide parallel RANSAC toward more efficient sampling. Surface normal calculation is also accelerated with utilization of integral images. Experimental results using real data shows our approach is able to run in realtime and orders of magnitude faster than traditional RANSAC while achieving similar precision.

"Parallel RANSAC: Speeding up plane extraction in RGBD image sequences using GPU", 5th International Conference on Computer and Knowledge Engineering, Oct, 2015, Mashhad, Iran.