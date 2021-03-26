#include "right_looking.cu"

float * copyDataToDevice (float *d_data, float *h_data, int totalElements) {

    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&d_data, totalElements);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector d_data (error code %s)!\n",
             cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copying data from host memory to the CUDA device\n");
    err = cudaMemcpy(d_data, h_data, totalElements, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy h_data from host to device (error code %s)\n", 
             cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    printf("Copied data successfully!\n");
    
    return d_data;
}

int main()
{
    FILE *fptr;
    fptr = fopen("input.txt", "r");
    int n, dim;
    // char str[50];
    int temp;
    fscanf(fptr, "%d", &n);
    fscanf(fptr, "%d", &dim);
    float h_A[n*dim*dim];
    int count = 0;
    int x = 0;
    int gidx = 0;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            for (int k = 0; k < dim; k++)
            {
                fscanf(fptr, "%d", &temp);
                x = j * dim + k;
                gidx = x * n + i;
                h_A[gidx] = temp;
            }
        }
    }
    int size = n * dim * dim;
    
    float *d_A = NULL;
    float * read_data = copyDataToDevice(d_A, h_A, size);
    int N = dim, i, j;
//     float M[n*dim*dim] = h_A;
    printf("Testing for matrix M [%dx%d]\n",N,N);
    dim3 grid(1,1,1);
    dim3 block(TILE_SIZE,TILE_SIZE,1);
    right_looking_launch_kernel<<<grid,block>>>(read_data,N);
    cudaError_t err = cudaMemcpy(h_A,read_data,size,cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy the output matrix M from device to Host (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Printing output matrix\n");
    for(i=0;i<n;i++)
    {
        for(j=0;j<n;j++)
        {
            if(j<=i)
                printf("%f\t",h_A[i*N + j]);
            else
                printf("%f\t",0.0);
        }
        printf("\n");
    }
    err = cudaFree(read_data);
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix M (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceReset();
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the CUDA device (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    free(h_A);
    printf("DONE!\n");
    return 0;
}
