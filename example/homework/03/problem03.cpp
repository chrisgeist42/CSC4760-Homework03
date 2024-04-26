#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 100 // Total number of elements in y

int main(int argc, char *argv[]) {
    int rank, size;
    int P, Q;
    int local_M;
    int i, j;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    P = size;    
    Q = 1;

    if (rank == 0) {
        printf("P = %d, Q = %d\n", P, Q);
    }

    int dims[2] = {P, Q};
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    if (coords[1] == 0) {
        if (coords[0] == 0) {
            int *x = (int *)malloc(N * sizeof(int));
            for (i = 0; i < N; i++) {
                x[i] = i; 
	    }
            local_M = N / P;
            int *local_x = (int *)malloc(local_M * sizeof(int));
            MPI_Scatter(x, local_M, MPI_INT, local_x, local_M, MPI_INT, 0, MPI_COMM_WORLD);
            free(x);
        } else {
            local_M = N / P;
            int *local_x = (int *)malloc(local_M * sizeof(int));
            MPI_Scatter(NULL, local_M, MPI_INT, local_x, local_M, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }
    int J = coords[1] * P + coords[0];
    j = J / Q;
    int q = J % Q;
    int elements_per_process = N / Q;
    int start_index = q * elements_per_process;
    int *y = (int *)malloc(elements_per_process * sizeof(int));

    MPI_Bcast(y, elements_per_process, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Process (%d, %d) - y:", coords[0], coords[1]);
    for (i = 0; i < elements_per_process; i++) {
        printf(" %d", y[i]);
    }
    printf("\n");

    MPI_Finalize();
    free(y);

    return 0;
}
