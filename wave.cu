/**********************************************************************
 * DESCRIPTION:
 *   Serial Concurrent Wave Equation - C Version
 *   This program implements the concurrent wave equation
 *********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAXPOINTS 1000000
#define MAXSTEPS 1000000
#define MINPOINTS 20
#define PI 3.14159265
#define Num 32

void check_param(void);
void init_line(void);
void update (void);
void printfinal (void);

int nsteps,                 	/* number of time steps */
    tpoints, 	     		/* total points along string */
    rcode;                  	/* generic return code */
float  values[MAXPOINTS+2], 	/* values at time t */
       oldval[MAXPOINTS+2], 	/* values at time (t-dt) */
       newval[MAXPOINTS+2]; 	/* values at time (t+dt) */


/**********************************************************************
 *	Checks input values from parameters
 *********************************************************************/
void check_param(void)
{
   char tchar[20];

   /* check number of points, number of iterations */
   while ((tpoints < MINPOINTS) || (tpoints > MAXPOINTS)) {
      printf("Enter number of points along vibrating string [%d-%d]: "
           ,MINPOINTS, MAXPOINTS);
      scanf("%s", tchar);
      tpoints = atoi(tchar);
      if ((tpoints < MINPOINTS) || (tpoints > MAXPOINTS))
         printf("Invalid. Please enter value between %d and %d\n", 
                 MINPOINTS, MAXPOINTS);
   }
   while ((nsteps < 1) || (nsteps > MAXSTEPS)) {
      printf("Enter number of time steps [1-%d]: ", MAXSTEPS);
      scanf("%s", tchar);
      nsteps = atoi(tchar);
      if ((nsteps < 1) || (nsteps > MAXSTEPS))
         printf("Invalid. Please enter value between 1 and %d\n", MAXSTEPS);
   }

   printf("Using points = %d, steps = %d\n", tpoints, nsteps);

}

/**********************************************************************
 *     Initialize points on line
 *********************************************************************/
/*void init_line(void)
{
   int i, j;
   float x, fac, k, tmp;

   // Calculate initial values based on sine curve 
   fac = 2.0 * PI;
   k = 0.0; 
   tmp = tpoints - 1;
   for (j = 1; j <= tpoints; j++) {
      x = k/tmp;
      values[j] = sin (fac * x);
      k = k + 1.0;
   } 

   //Initialize old values array
   for (i = 1; i <= tpoints; i++) 
      oldval[i] = values[i];
}*/

/**********************************************************************
 *      Calculate new values using wave equation
 *********************************************************************/
void do_math(int i)
{
   float dtime, c, dx, tau, sqtau;

   dtime = 0.3;
   c = 1.0;
   dx = 1.0;
   tau = (c * dtime / dx);
   sqtau = tau * tau;
   newval[i] = (2.0 * values[i]) - oldval[i] + (sqtau *  (-2.0)*values[i]);
}

/**********************************************************************
 *     Update all values along line a specified number of times
 *********************************************************************/
__global__ void update(float *values_out, int tpoints, int nsteps)
{
   int i;
   int j = 1+threadIdx.x;
   int idx = j+blockIdx.x*Num;

   if(idx<=tpoints){
       float values;
       float newval;
       float oldval;
       float x, fac, tmp;
       fac = 2.0*PI;
       tmp = tpoints-1;
       x=(float)(idx-1)/tmp;
       values = sin(fac*x);
       oldval = values;
       /* Update values for each time step */
       for (i = 1; i<= nsteps; i++) {
          if((idx==1) || (idx==tpoints))
              newval = 0.0;
          else
	      newval = (2.0*values)-oldval+(0.09*(-2.0*values));
          oldval = values;
	  values = newval;
       }
       values_out[idx] = values;
    }
}

/**********************************************************************
 *     Print final results
 *********************************************************************/
void printfinal()
{
   int i;

   for (i = 1; i <= tpoints; i++) {
      printf("%6.4f ", values[i]);
      if (i%10 == 0)
         printf("\n");
   }
}

/**********************************************************************
 *	Main program
 *********************************************************************/
int main(int argc, char *argv[])
{
	sscanf(argv[1],"%d",&tpoints);
	sscanf(argv[2],"%d",&nsteps);
        int size = (1+tpoints)*sizeof(float);
	check_param();
        float *values_dev;
        cudaMalloc((void**)&values_dev, size);
	printf("Initializing points on the line...\n");
	//init_line();
	printf("Updating all points for all time steps...\n");
	if(tpoints%Num){
            update<<<1+tpoints/Num, Num>>>(values_dev, tpoints, nsteps);
        }else{
            update<<<tpoints/Num, Num>>>(values_dev, tpoints, nsteps);
        }

        cudaMemcpy(values, values_dev, size, cudaMemcpyDeviceToHost);
	cudaFree(values_dev);

	printf("Printing final results...\n");
	printfinal();
	printf("\nDone.\n\n");
	return 0;
}
