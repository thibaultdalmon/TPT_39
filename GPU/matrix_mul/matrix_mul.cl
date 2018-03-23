__kernel void matrix_mul(__global const float *x,__global const float *y,__global float *restrict z)
{
	int index_row=get_global_id(0);
	int index_clmn=get_global_id(1);
	int index_var=get_global_id(2);
	int N=get_global_size(0);
	z[N*N*index_row+N*index_clmn+index_var]=x[N*index_row+index_var]*y[N*index_var+index_clmn];
}
