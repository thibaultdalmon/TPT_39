#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <fstream>
#include <time.h>
#include "opencv2/opencv.hpp"
#include <CL/cl.h>
#include <CL/cl_ext.h>
#define STRING_BUFFER_LEN 1024

using namespace cv;
using namespace std;
#define SHOW

enum {gauss, vert, hor};

void print_clbuild_errors(cl_program program,cl_device_id device);

void callback(const char *buffer, size_t length, size_t final, void *user_data);

void checkError(int status, const char *msg);

unsigned char ** read_file(const char *name);

void GPU_GaussianBlur(float * input, float * output,
                    int sizeX, int sizeY, int mode);

int status;
char char_buffer[STRING_BUFFER_LEN];
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_context_properties context_properties[] =
{
	CL_CONTEXT_PLATFORM, 0,
    CL_PRINTF_CALLBACK_ARM, (cl_context_properties)callback,
    CL_PRINTF_BUFFERSIZE_ARM, 0x10000,
    0
};
cl_command_queue queue;
cl_program program;
cl_kernel kernel;
cl_mem input_a_buf; // num_devices elements
cl_mem input_b_buf; // num_devices elements
cl_mem output_buf; // num_devices elements
cl_event write_event[2];
cl_event kernel_event,finish_event;

int main(int, char**)
{

	//Kernel_init
    clGetPlatformIDs(1, &platform, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

    context_properties[1] = (cl_context_properties)platform;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);
    cl_event write_event[2];
    cl_event kernel_event,finish_event;

    unsigned char **opencl_program=read_file("filter.cl");
    program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
    if (program == NULL)
    {
         printf("Program creation failed\n");
         return -1;
    }
    int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
    kernel = clCreateKernel(program, "filter", NULL);

	//OpenCV_init
    VideoCapture camera("./bourne.mp4");
    if(!camera.isOpened())  // check if we succeeded
        return -1;

    const string NAME = "./output1.avi";   // Form the new name with container
    int ex = static_cast<int>(CV_FOURCC('M','J','P','G'));
    Size S = Size((int) camera.get(CV_CAP_PROP_FRAME_WIDTH),
		 		(int) camera.get(CV_CAP_PROP_FRAME_HEIGHT));
	//Size S =Size(1280,720);

    VideoWriter outputVideo;                                        // Open the output
        outputVideo.open(NAME, ex, 25, S, true);

    if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << NAME << endl;
        return -1;
    }
	time_t start,end;
	double diff,tot;
	int count=0;
	tot = 0;
	const char *windowName = "filter";   // Name shown in the GUI window.
    #ifdef SHOW
    namedWindow(windowName); // Resizable window, might not work on Windows.
    #endif
	
	int Nx = (int) camera.get(CV_CAP_PROP_FRAME_HEIGHT)+2;
	int Ny = (int) camera.get(CV_CAP_PROP_FRAME_WIDTH)+2;
	// Input buffers.
    input_a_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
        Nx*Ny* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    input_b_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
        3*3*sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    // Output buffer.
    output_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
        Nx*Ny* sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");
	
    while (true) {
        Mat cameraFrame,displayframe;
		count=count+1;
		if(count > 299) break;
        camera >> cameraFrame;
		time (&start);
        Mat filterframe = Mat(cameraFrame.size(), CV_32FC3);
        Mat grayframe,edge_x,edge_y,edge;
    	cvtColor(cameraFrame, grayframe, CV_BGR2GRAY);
		
		//Mycode
		int *p = grayframe.size.p;
		int dim = grayframe.dims;
		int tmp = 1;
		for (int i = 0; i<dim; i++){
			tmp *= p[i];
		}

		grayframe.convertTo(edge_x, CV_32FC1);
		grayframe.convertTo(edge_y, CV_32FC1);
		grayframe.convertTo(grayframe, CV_32FC1);

		//printf("Debut GaussianBlur, p[0]=%d, p[1]=%d, tmp=%d\n", p[0], p[1], tmp);
		
		GPU_GaussianBlur((float*)grayframe.data, (float*)grayframe.data, p[0], p[1], gauss);
		GPU_GaussianBlur((float*)grayframe.data, (float*)grayframe.data, p[0], p[1], gauss);
		GPU_GaussianBlur((float*)grayframe.data, (float*)grayframe.data, p[0], p[1], gauss);
		
		
		grayframe.convertTo(edge_x, CV_32FC1);
		grayframe.convertTo(edge_y, CV_32FC1);
		GPU_GaussianBlur((float*)edge_x.data, (float*)edge_x.data, p[0], p[1], vert);
		GPU_GaussianBlur((float*)edge_y.data, (float*)edge_y.data, p[0], p[1], hor);
			
		//printf("Fin GaussianBlur\n");	
	
		edge_x.convertTo(edge_x, CV_8UC1);
		edge_y.convertTo(edge_y, CV_8UC1);
		//EndofMycode

		/*	
    	GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
    	GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
    	GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
		Scharr(grayframe, edge_x, CV_8U, 0, 1, 1, 0, BORDER_DEFAULT );
		Scharr(grayframe, edge_y, CV_8U, 1, 0, 1, 0, BORDER_DEFAULT );*/
		
		addWeighted( edge_x, 0.5, edge_y, 0.5, 0, edge );
        threshold(edge, edge, 80, 255, THRESH_BINARY_INV);
		time (&end);
        cvtColor(edge, edge_inv, CV_GRAY2BGR);
    	// Clear the output image to black, so that the cartoon line drawings will be black (ie: not drawn).
    	memset((char*)displayframe.data, 0, displayframe.step * displayframe.rows);
		grayframe.copyTo(displayframe,edge);
        cvtColor(displayframe, displayframe, CV_GRAY2BGR);
		outputVideo << displayframe;
	#ifdef SHOW
        imshow(windowName, displayframe);
	#endif
		diff = difftime (end,start);
		tot+=diff;
	}
	outputVideo.release();
	camera.release();
  	printf ("FPS %.2lf .\n", 299.0/tot );

	clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseMemObject(input_a_buf);
    clReleaseMemObject(input_b_buf);
    clReleaseMemObject(output_buf);
    clReleaseCommandQueue(queue);	

    return EXIT_SUCCESS;

}

void print_clbuild_errors(cl_program program,cl_device_id device)
    {
        cout<<"Program Build failed\n";
        size_t length;
        char buffer[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
        cout<<"--- Build log ---\n "<<buffer<<endl;
        exit(1);
    }

unsigned char ** read_file(const char *name) {
  	size_t size;
  	unsigned char **output=(unsigned char **)malloc(sizeof(unsigned char *));
  	FILE* fp = fopen(name, "rb");
  	if (!fp) {
    	printf("no such file:%s",name);
    	exit(-1);
  	}

  	fseek(fp, 0, SEEK_END);
  	size = ftell(fp);
  	fseek(fp, 0, SEEK_SET);

  	*output = (unsigned char *)malloc(size);
  	if (!*output) {
    	fclose(fp);
    	printf("mem allocate failure:%s",name);
    	exit(-1);
  	}

  	if(!fread(*output, size, 1, fp)) printf("failed to read file\n");
  	fclose(fp);
  	return output;
}


void callback(const char *buffer, size_t length, size_t final, void *user_data)
{
     fwrite(buffer, 1, length, stdout);
}


void checkError(int status, const char *msg) {
    if(status!=CL_SUCCESS)
        printf("%s\n",msg);
}

// Randomly generate a floating-point number between -10 and 10.
float rand_float() {
  return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

void GPU_GaussianBlur(float * input, float * output, 
					int sizeX, int sizeY, int mode)
{
	//printf("sizeX = %d, sizeY= %d\n", sizeX, sizeY);

	int Nx = sizeX+2;
    int Ny = sizeY+2;

	//Matrix Preparation
	
	float input_a[Nx*Ny];
	float output_tmp[Nx*Ny];
	for (int i = 0; i<Nx; i++){
		for (int j = 0; j<Ny; j++){
			if ((!i)||(!j)){
				input_a[i*Ny+j] = 0;
			}else{
				if ((i>sizeX)||(j>sizeY)){
					input_a[i*Ny+j] = 0;
				}else{
					input_a[i*Ny+j] = input[(i-1)*sizeY+(j-1)];
				}
			}
		}
	}

	//printf("Fin init : sizeX=%u, sizeY=%u\n", Nx, Ny);
	float filter[3*3];
	switch (mode){
	case(gauss) :
		filter[0] = 1./16; 
		filter[1] = 2./16; 
		filter[2] = 1./16; 
		filter[3] = 2./16; 
		filter[4] = 4./16; 
		filter[5] = 2./16; 
		filter[6] = 1./16; 
		filter[7] = 2./16; 
		filter[8] = 1./16;
		break;
	case(vert) : 
		filter[0] = -3;
        filter[1] = 0;
        filter[2] = 3;
        filter[3] = -3;
        filter[4] = 0;
        filter[5] = 3;
        filter[6] = -3;
        filter[7] = 0;
        filter[8] = 3;
		break;
	default : 
		filter[0] = -3;		
        filter[1] = -3;
        filter[2] = -3;
        filter[3] = 0;
        filter[4] = 0;
        filter[5] = 0;
        filter[6] = 3;
        filter[7] = 3;
        filter[8] = 3;
	}
	//float filter[3*3] = {0,0,0,0,0,0,0,0,0};

    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    status = clEnqueueWriteBuffer(queue, input_a_buf, CL_TRUE,
        0, (Nx)*(Ny)* sizeof(float), input_a, 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer input A");

    status = clEnqueueWriteBuffer(queue, input_b_buf, CL_TRUE,
        0, 3*3* sizeof(float), filter, 0, NULL, &write_event[1]);
    checkError(status, "Failed to transfer input B");
	

	// Set kernel arguments.
    unsigned argi = 0;

    size_t global_work_size[2] = {(unsigned int) sizeX,(unsigned int) sizeY};
    size_t local_work_size[2] = {1,1};

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_a_buf);
    checkError(status, "Failed to set argument 1");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_b_buf);
    checkError(status, "Failed to set argument 2");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 3");


    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
            global_work_size, local_work_size, 2, write_event, &kernel_event);
    checkError(status, "Failed to launch kernel 1");

    // Read the result. This the final operation.
    status = clEnqueueReadBuffer(queue, output_buf, CL_TRUE,
        0, Nx*Ny* sizeof(float), output_tmp, 1, &kernel_event, &finish_event);
	checkError(status, "Reading Failure");

	for (int i=0; i<sizeX; i++){
		for (int j=0; j<sizeY; j++){
			output[i*sizeY+j] = output_tmp[(i+1)*Ny+j+1];
			//output[i*sizeY+j] = 255;
		}
	}
}
