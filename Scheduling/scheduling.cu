#include <stdio.h>
#include <stdint.h>

/*******************************/
/* STREAMING MULTIPROCESSOR ID */
/*******************************/
static __device__ __inline__ uint32_t __mysmid() {
	uint32_t smid;
	asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
	return smid;
}

/***********/
/* WARP ID */
/***********/
static __device__ __inline__ uint32_t __mywarpid() {
	uint32_t warpid;
	asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
	return warpid;
}

/****************/
/* WARP LANE ID */
/****************/
static __device__ __inline__ uint32_t __mylaneid() {
	uint32_t laneid;
	asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
	return laneid;
}

/*******************/
/* KERNEL FUNCTION */
/*******************/
__global__ void mykernel() {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	printf("Thread ID = %d;\t SM ID = %d;\t Warp ID = %d;\t Warp lane = %d\n", idx, __mysmid(), __mywarpid(), __mylaneid());
}

/********/
/* MAIN */
/********/
int main() {

	mykernel << <2, 64 >> >();
	cudaDeviceSynchronize();

	return 0;

}
