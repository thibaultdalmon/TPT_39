__kernel void average(__global const float *x,__global float *restrict z)
{
	int index_row=get_group_id(0);
	int index_clmn=get_local_id(0);
	int N = get_local_size(0);
	if(!index_clmn){
		for (int i = 0; i<N; i++){
			z[index_row]+=x[index_row*N+i];
		}
	}
}
