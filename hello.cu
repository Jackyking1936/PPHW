#include <stdio.h>
#define N 512

__global__ void add(int *a, int *b, int *c){
	c[blockIdx.x] = a[blockIdx.x]+b[blockIdx.x];
}

void random_ints(int *a, int n){
	for(int i=0; i<n; i++){
		a[i] = rand()%10+1;
	}
}

int main(void){

	int *a,*b,*c;
	int *d_a,*d_b,*d_c;
	int size = N*sizeof(int);

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	a=(int*)malloc(size); random_ints(a, N);
	b=(int*)malloc(size); random_ints(b, N);
	c=(int*)malloc(size);

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	add<<<N,1>>>(d_a, d_b, d_c);

	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	
	for(int i=0; i<N; i++)
		printf("Hello World!%d %d %d\n", a[i], b[i], c[i]);
	
	free(a); free(b);free(c);
	return 0;
}
