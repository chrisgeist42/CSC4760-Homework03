#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define M 100

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

            int *x = (int *)malloc(M * sizeof(int));
            for (i = 0; i < M; i++) {
                x[i] = i;
            }
            local_M = M / P;

            int *local_x = (int *)malloc(local_M * sizeof(int));
            MPI_Scatter(x, local_M, MPI_INT, local_x, local_M, MPI_INT, 0, MPI_COMM_WORLD);
            free(x);
        } else {
            local_M = M / P;
            int *local_x = (int *)malloc(local_M * sizeof(int));

            MPI_Scatter(NULL, local_M, MPI_INT, local_x, local_M, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    int *y = (int *)malloc(local_M * sizeof(int));

    MPI_Bcast(y, local_M, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, y, local_M, MPI_INT, MPI_SUM, cart_comm);

    for (i = 0; i < P; i++) {
        if (coords[0] == i) {
            printf("Process (%d, %d) - y: ", coords[0], coords[1]);
            for (j = 0; j < local_M; j++) {
                printf("%d ", y[j]);
            }
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    free(y);

    return 0;
}
