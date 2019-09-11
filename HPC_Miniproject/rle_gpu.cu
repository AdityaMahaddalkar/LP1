
__global__ void maskKernel(int *g_in, int *g_backwardMask, int n) {
	
	for(int i: hemi::grid_stride_range(0, n)){
		if(i == 0)
			g_backwardMask[i] = 1;
		else{
			g_backwardMask[i] = (g_in[i] != g_in[i - 1])
		}
	}
}

__global__ void compactKernel(int *g_scannedBackwardMask,
								int *g_compactedBackwardMask,
								int *g_totalRuns,
								int n)
{

	for(int i : hemi::grid_stride_range(0, n)){

		if(i == (n-1)){
			g_compactedBackwardMask[g_scannedBackwardMask[i]] = i + 1;
		}

		if(i == 0){
			g_compactedBackwardMask[0] = 0;
		}

		else if(g_scannedBackwardMask[i] != g_scannedBackwardMask[i - 1]){
			g_compactedBackwardMask[g_scannedBackwardMask[i] - 1] = i
		}

	}

}


__global__ void scatterKernel(int *g_compactedBackwardMask,
								int *g_totalRuns,
								int *g_in,
								int *g_symbolsOut,
								int *g_countsOut)
{
	
	int n = *g_totalRuns;

	for(int i : hemi::grid_stride_range(0, n)){
		int a = g_compactedBackwardMask[i];
		int b = g_compactedBackwardMask[i + 1];

		g_symbolsOut[i] = g_in[a];
		g_countsOut[i] = b - a;
	}
}