#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define M 100

int main(int argc, char *argv[]) {
    int rank, size;
    int P, Q;
    int local_M;
    int i, j;
    int local_dot_product = 0;
    int global_dot_product = 0;

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

    int *x = (int *)malloc(M * sizeof(int));
    if (coords[1] == 0) {
        local_M = M / P;
        int *local_x = (int *)malloc(local_M * sizeof(int));
        if (coords[0] == 0) {
            for (i = 0; i < M; i++) {
                x[i] = i;
            }
            MPI_Scatter(x, local_M, MPI_INT, local_x, local_M, MPI_INT, 0, MPI_COMM_WORLD);
        } else {
            MPI_Scatter(NULL, local_M, MPI_INT, local_x, local_M, MPI_INT, 0, MPI_COMM_WORLD);
        }
        free(local_x);
    }

    int *y = (int *)malloc(M * sizeof(int));
    if (coords[0] == 0) {
        if (coords[1] == 0) {
            for (i = 0; i < M; i++) {
                y[i] = i * size + rank;
            }
        }
        MPI_Scatter(y, local_M, MPI_INT, y, local_M, MPI_INT, 0, cart_comm);
    }
    free(y);

    MPI_Bcast(x, M, MPI_INT, 0, cart_comm);

    local_dot_product = 0;
    for (i = 0; i < local_M; i++) {
        local_dot_product += x[i] * y[i];
    }

    MPI_Reduce(&local_dot_product, &global_dot_product, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Global dot product: %d\n", global_dot_product);
    }

    MPI_Finalize();
    free(x);

    return 0;
}
