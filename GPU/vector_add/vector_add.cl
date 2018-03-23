__kernel void vector_add(__global const float *x,
						__global const int *k,  
                        __global float *restrict z)
{

 	int index = get_group_id(0);
	int index2 = get_local_id(0);
	int max = get_global_size(0);
	printf("group id : %d, %d\n", get_local_size(0), get_global_size(0));
	
	if (index2){
		if ((2*index+index2)*k[0]<max){
			z[2*index*k[0]] = x[2*index*k[0]] + x[(2*index+index2)*k[0]];
			printf("value : %f, %d, %d\n", z[2*index*k[0]], 2*index*k[0], 2*index+index2*k[0]);
		}
		else{
			if (2*index*k[0]){
				z[2*index*k[0]] = x[2*index*k[0]];
			}
		}
	}

}

