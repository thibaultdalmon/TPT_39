__kernel void filter(__global const float *x,
					__global const float *y,
					__global float *restrict z)
{
	int index_row = get_global_id(0);
	int index_clmn = get_global_id(1);
	int N = get_global_size(1);

	z[(index_row+1)*N+index_clmn+1] = 0;
	for (int i = 0; i<3; i++){
		for (int j = 0; j<3 ; j++){
			z[(index_row+1)*N+index_clmn+1]+=x[(index_row+i)*N+(index_clmn+j)]*y[i*3+j];
		}
	}
}






