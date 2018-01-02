#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <CL/cl.h>
#define MAX_SOURCE_SIZE (0x100000)


using namespace std;


int main(int argc, char** argv)
{

    FILE *fp;
    size_t source_size;
    char *ProgramSource;
    const char filename[]="./histogram.cl";	

    fp=fopen(filename,"r");
    ProgramSource=(char *)malloc(MAX_SOURCE_SIZE);
    source_size=fread(ProgramSource,1,MAX_SOURCE_SIZE,fp);
    fclose(fp);


    cl_int err;                         

    size_t global;                     
    size_t local;                      
    size_t max_work[3];
    size_t max_items;

    cl_device_id device_id;             
    cl_context context;                
    cl_command_queue commands;          
    cl_program program;                 
    cl_kernel kernel;                   
    cl_platform_id platform;
    
    unsigned int *histogram_results = new unsigned int[256*3];
    unsigned int i=0, a, input_size;
    unsigned int bound;
    unsigned int part_size;

    fstream inFile("input", ios_base::in);
    ofstream outFile("0653409.out", ios_base::out);
    
    memset(histogram_results, 0, sizeof(unsigned int)*256*3);
    
    inFile >> input_size;
    bound = input_size/3;

    unsigned int *image = new unsigned int[input_size];
    while( inFile >> a ) {
	image[i++] = a;
    }
   
    clGetPlatformIDs(1, &platform, NULL); 
   
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work), &max_work, NULL);
    
    max_items = max_work[0]*max_work[1]*max_work[2];

    if(bound > max_items) { 
        part_size = bound / max_items;
        if(bound%max_items!=0) part_size++;
    }
    else { 
        part_size = 1;
    }
  
 
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    commands = clCreateCommandQueue(context, device_id, 0, &err);

    cl_mem img_in = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(unsigned int)*input_size, NULL, NULL);
    cl_mem his_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned int)*256*3, NULL, NULL);
    program = clCreateProgramWithSource(context, 1, (const char **) &ProgramSource, NULL, &err);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    kernel = clCreateKernel(program, "histogram", &err);
    err = clEnqueueWriteBuffer(commands, img_in, CL_TRUE, 0, sizeof(unsigned)*input_size, image, 0, NULL, NULL);
    	
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &img_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &his_out);
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &bound);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &part_size);

    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);

    global = max_items;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);

    clFinish(commands);

    err = clEnqueueReadBuffer(commands, his_out, CL_TRUE, 0, sizeof(unsigned int)*256*3, histogram_results, 0, NULL, NULL );
 	
    for(unsigned int i = 0; i < 256 * 3; ++i) {
        if (i%256 == 0 && i!=0) outFile << endl;
        outFile << histogram_results[i]<< ' ';
    }
	
    delete [] histogram_results;
    delete [] image;
    inFile.close();
    outFile.close();
    
    clReleaseMemObject(img_in);
    clReleaseMemObject(his_out);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}
