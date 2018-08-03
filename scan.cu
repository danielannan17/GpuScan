#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define CUDA_ERROR( err, msg ) { \
if (err != cudaSuccess) {\
printf( "%s: %s in %s at line %d\n", msg, cudaGetErrorString( err ), __FILE__, __LINE__);\
exit( EXIT_FAILURE );\
}\
}

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
 ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

#define BLOCK_SIZE 1024
size_t size = 1000000;

void sequential_block_scan(int * x, int * y, int len) {
	int num_blocks = 1 + (len - 1) / BLOCK_SIZE;
	for (int blk = 0; blk < num_blocks; blk++) {
		int blk_start = blk * BLOCK_SIZE;
		int blk_end = blk_start + BLOCK_SIZE;
		if (blk_end > len)
			blk_end = len;
		y[blk_start] = 0;
		for (int i = blk_start + 1; i < blk_end; i++)
			y[i] = y[i - 1] + x[i - 1];

	}
}

__device__ void seq() {

}

void sequential_full_scan(int * x, int * y, int len) {
	y[0] = 0;
	for (int i = 1; i < len; i++)
		y[i] = y[i - 1] + x[i - 1];
}

static void compare_results(const int *vector1, const int *vector2,
		int numElements) {
	for (int i = 0; i < numElements; ++i) {
		if (fabs(vector1[i] - vector2[i]) > 0) {

			for (int x = max(i - 5, 0); x < max(i - 5, 0) + 10; x++) {
				if (x == i)
					printf(" | ");
				printf("(%d,%d)", vector1[x], vector2[x]);
			}
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}
}

__global__ void blockScan(int * g_odata, int * g_idata, int n, int * sums) {
	__shared__ int temp[2 * BLOCK_SIZE]; // allocated on invocation
	int blockStart = 2 * BLOCK_SIZE * blockIdx.x;
	int thid = threadIdx.x;
	int offset = 1;

	if (2 * thid + blockStart < n) {
		temp[2 * thid] = g_idata[2 * thid + blockStart]; // load input into shared memory
		temp[2 * thid + 1] = g_idata[2 * thid + 1 + blockStart];
	}

	for (int d = 2 * BLOCK_SIZE >> 1; d > 0; d >>= 1) // build sum in place up the tree
			{
		__syncthreads();
		if (thid < d) {
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (sums != NULL) {
		sums[blockStart / BLOCK_SIZE] = temp[BLOCK_SIZE - 1];
		sums[(blockStart / BLOCK_SIZE) + 1] = temp[2 * BLOCK_SIZE - 1]
				- temp[BLOCK_SIZE - 1];
	}
	__syncthreads();

	if (thid == 0) {
		temp[BLOCK_SIZE - 1] = 0;
	}

	if (BLOCK_SIZE % thid == 0) {
		temp[2 * BLOCK_SIZE - 1] = 0;
	}

	for (int d = 1; d < 2 * BLOCK_SIZE; d *= 2) // traverse down tree & build scan
			{
		offset >>= 1;

		__syncthreads();
		if (thid < d) {
			int ai = offset * (2 * thid + 1) - 1; //1024
			int bi = offset * (2 * thid + 2) - 1; //1025
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;

		}
	}

	if (2 * thid + blockStart < n) {
		g_odata[(2 * thid) + blockStart] = temp[2 * thid]; // write results to device memory
		g_odata[(2 * thid + 1) + blockStart] = temp[2 * thid + 1];
	}
	__syncthreads();
}

__global__ void blockScanWithBCAO(int * g_odata, int * g_idata, int n,
		int * sums) {
	__shared__ int temp[4 * BLOCK_SIZE]; // allocated on invocation
	int blockStart = 2 * BLOCK_SIZE * blockIdx.x;
	int thid = threadIdx.x;
	int offset = 1;
	int ai = thid;
	int bi = thid + (BLOCK_SIZE / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	if (ai + bankOffsetA + blockStart < n) {
		temp[ai + bankOffsetA] = g_idata[ai + blockStart];
		temp[bi + bankOffsetB] = g_idata[bi + blockStart];
	}

	for (int d = 2 * BLOCK_SIZE >> 1; d > 0; d >>= 1) // build sum in place up the tree
			{
		__syncthreads();
		if (thid < d) {
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (sums != NULL) {
		sums[blockStart / BLOCK_SIZE] = temp[BLOCK_SIZE - 1 + CONFLICT_FREE_OFFSET(BLOCK_SIZE - 1)];
		sums[(blockStart / BLOCK_SIZE) + 1] = temp[2*BLOCK_SIZE - 1 + CONFLICT_FREE_OFFSET(2*BLOCK_SIZE - 1)]
				- temp[BLOCK_SIZE - 1 + CONFLICT_FREE_OFFSET(BLOCK_SIZE - 1)];
	}
	__syncthreads();

	if (thid == 0) {
		temp[BLOCK_SIZE - 1 + CONFLICT_FREE_OFFSET(BLOCK_SIZE - 1)] = 0;
	}

	if (BLOCK_SIZE % thid == 0) {
		temp[2*BLOCK_SIZE - 1 + CONFLICT_FREE_OFFSET(2*BLOCK_SIZE - 1)] = 0;
	}


	for (int d = 1; d < 2 * BLOCK_SIZE; d *= 2) // traverse down tree & build scan
			{
		offset >>= 1;

		__syncthreads();
		if (thid < d) {
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;

		}
	}
	if (ai + blockStart < n) {
		g_odata[ai  + blockStart] = temp[ai + bankOffsetA];
		g_odata[bi  + blockStart] = temp[bi + bankOffsetB];
	}
	__syncthreads();
}

__global__ void addToBlock2(int * input, int * output, int n) {
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	int numBlocks = 1 + ((n - 1) / (BLOCK_SIZE));

	if (thid < numBlocks) {
		int blockStart = BLOCK_SIZE * thid;
		for (int i = 0; i < BLOCK_SIZE; i++) {
			output[i + blockStart] += input[thid];
		}
	}
	__syncthreads();

}

void full_scan_without_BCAO(int * g_odata, int * g_idata, int n) {
	int blocksPerGrid = 1 + ((n - 1) / (BLOCK_SIZE * 2));
	cudaError_t err = cudaSuccess;
	int lvl3 = 0;
	if (n / BLOCK_SIZE > BLOCK_SIZE) {
		lvl3 = 1;
	}

	int *d_sum1 = NULL;
	err = cudaMalloc((void **) &d_sum1, ((n / BLOCK_SIZE) + 1) * sizeof(int));

	CUDA_ERROR(err, "Failed to allocate for d_sum1");

	int *d_sum1_scanned = NULL;
	err = cudaMalloc((void **) &d_sum1_scanned,
			((n / BLOCK_SIZE) + 1) * sizeof(int));
	CUDA_ERROR(err, "Failed to allocate for d_sum1_scanned");

	if (lvl3 == 0) {
		blockScan<<<blocksPerGrid, BLOCK_SIZE>>>(g_odata, g_idata, n, d_sum1);
		blockScan<<<blocksPerGrid, BLOCK_SIZE>>>(d_sum1_scanned, d_sum1,
				(n / BLOCK_SIZE) + 1, NULL);
		addToBlock2<<<blocksPerGrid, BLOCK_SIZE>>>(d_sum1_scanned, g_odata, n);
		cudaDeviceSynchronize();
	} else {
//Level 3 Scan Needed
		int *d_sum2 = NULL;
		err = cudaMalloc((void **) &d_sum2,
				((n / BLOCK_SIZE) + 1) * sizeof(int));
		CUDA_ERROR(err, "Failed to allocate for d_sum1");
		int *d_sum2_scanned = NULL;
		err = cudaMalloc((void **) &d_sum2_scanned,
				((n / BLOCK_SIZE) + 1) * sizeof(int));
		CUDA_ERROR(err, "Failed to allocate for d_sum1");
		blockScan<<<blocksPerGrid, BLOCK_SIZE>>>(g_odata, g_idata, n, d_sum1);
		cudaDeviceSynchronize();

		err = cudaGetLastError();
		CUDA_ERROR(err, "1");

		blockScan<<<(blocksPerGrid / (BLOCK_SIZE * 2)), BLOCK_SIZE>>>(
				d_sum1_scanned, d_sum1, (n / BLOCK_SIZE) + 1, d_sum2);
		cudaDeviceSynchronize();

		err = cudaGetLastError();
		CUDA_ERROR(err, "2");

		blockScan<<<1, BLOCK_SIZE>>>(d_sum2_scanned, d_sum2, 1, NULL);
		cudaDeviceSynchronize();
		cudaDeviceSynchronize();

		err = cudaGetLastError();
		CUDA_ERROR(err, "3");
		addToBlock2<<<1 + (blocksPerGrid / BLOCK_SIZE * 2), BLOCK_SIZE>>>(
				d_sum2_scanned, d_sum1_scanned, ((n / BLOCK_SIZE) + 1));
		cudaDeviceSynchronize();

		err = cudaGetLastError();
		CUDA_ERROR(err, "4");
		addToBlock2<<<blocksPerGrid, BLOCK_SIZE>>>(d_sum1_scanned, g_odata, n);
		cudaDeviceSynchronize();

		err = cudaGetLastError();
		CUDA_ERROR(err, "5");
		cudaDeviceSynchronize();
	}

}

void block_scan_without_BCAO(int * g_odata, int * g_idata, int n) {
	int blocksPerGrid = 1 + ((size - 1) / (BLOCK_SIZE * 2));
	blockScan<<<blocksPerGrid, BLOCK_SIZE>>>(g_odata, g_idata, n, NULL);
}

void block_scan_with_BCAO(int * g_odata, int * g_idata, int n) {
	int blocksPerGrid = 1 + ((size - 1) / (BLOCK_SIZE * 2));
	blockScanWithBCAO<<<blocksPerGrid, BLOCK_SIZE>>>(g_odata, g_idata, n, NULL);

}

int main(void) {

	cudaError_t err = cudaSuccess;

	StopWatchInterface * timer = NULL;
	sdkCreateTimer(&timer);
	double h_msecs;

	// Create Device timer event objects
	cudaEvent_t start, stop;
	float d_msecs;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Generate array
	int *h_inputVector = (int *) malloc(size * sizeof(int));
	memset(h_inputVector, '\0', size * sizeof(int));
	for (int i = 0; i < size; i++) {
		h_inputVector[i] = rand() % 10;
	};

	//Serial Block Scan
	int *h_outputBlockScan = (int *) malloc(size * sizeof(int));
	sdkStartTimer(&timer);
	sequential_block_scan(h_inputVector, h_outputBlockScan, size);
	sdkStopTimer(&timer);
	h_msecs = sdkGetTimerValue(&timer);

	//Serial Full Scan
	int *h_outputFullScan = (int *) malloc(size * sizeof(int));
	sdkStartTimer(&timer);
	sequential_full_scan(h_inputVector, h_outputFullScan, size);
	sdkStopTimer(&timer);
	h_msecs = sdkGetTimerValue(&timer);

	//Copying input vector to device
	int *d_inputVector = NULL;
	err = cudaMalloc((void **) &d_inputVector, (size * sizeof(int)));
	CUDA_ERROR(err, "Failed to allocate device vector");

	err = cudaMemcpy(d_inputVector, h_inputVector, size * sizeof(int),
			cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy vector from host to device");

	//Parallel Block Scan Without BCAO
	int *d_parallelBlockScan = NULL;
	err = cudaMalloc((void **) &d_parallelBlockScan, size * sizeof(int));
	CUDA_ERROR(err, "Failed to allocate for parallel output");

	cudaEventRecord(start, 0);
	block_scan_without_BCAO(d_parallelBlockScan, d_inputVector, size);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// wait for device to finish
	cudaDeviceSynchronize();

	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch vectorAdd kernel");

	err = cudaEventElapsedTime(&d_msecs, start, stop);
	CUDA_ERROR(err, "Failed to get elapsed time");
	int *h_parallelBlockScan = (int *) malloc(size * sizeof(int));
	err = cudaMemcpy(h_parallelBlockScan, d_parallelBlockScan,
			size * sizeof(int), cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy parallel output from device to host");
	compare_results(h_outputBlockScan, h_parallelBlockScan, size);
	printf("Block Scan Without BCAO = %.5fmSecs\n", d_msecs);




























	//Full Block Scan Without BCAO
	int *d_parallelFullScan = NULL;
	err = cudaMalloc((void **) &d_parallelFullScan, size * sizeof(int));
	CUDA_ERROR(err, "Failed to allocate for d_parallelFullScan");

	cudaEventRecord(start, 0);
	full_scan_without_BCAO(d_parallelFullScan, d_inputVector, size);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	// wait for device to finish
	cudaDeviceSynchronize();

	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch full_scan kernel");

	err = cudaEventElapsedTime(&d_msecs, start, stop);
	CUDA_ERROR(err, "Failed to get elapsed time");
	int *h_parallelFullScan = (int *) malloc(size * sizeof(int));
	err = cudaMemcpy(h_parallelFullScan, d_parallelFullScan, size * sizeof(int),
			cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy parallel output from device to host");
	compare_results(h_outputFullScan, h_parallelFullScan, size);
	printf("Full Scan Without BCAO = %.5fmSecs\n", d_msecs);

}
