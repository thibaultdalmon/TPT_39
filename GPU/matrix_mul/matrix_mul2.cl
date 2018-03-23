__kernel void matrix_mul2(__global const float *x,__global float *restrict z)
{
 	int index_row=get_local_id(0);
	int index_clmn=get_local_id(1);
	int N=get_global_size(0);
	int index_var=get_local_id(2);
	mem_fence(CLK_GLOBAL_MEM_FENCE);
	z[N*index_row+index_clmn]+=x[N*N*index_row+N*index_clmn+index_var];
}
