#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // setting P and Q
    int P, Q = 1;

    // color is rank / Q
    MPI_Comm split_comm1;
    int color1 = world_rank / Q;
    MPI_Comm_split(MPI_COMM_WORLD, color1, world_rank, &split_comm1);

    // color is rank mod Q
    MPI_Comm split_comm2;
    int color2 = world_rank % Q;
    MPI_Comm_split(MPI_COMM_WORLD, color2, world_rank, &split_comm2);

    // Clean up
    MPI_Comm_free(&split_comm1);
    MPI_Comm_free(&split_comm2);

    MPI_Finalize();
    return 0;
}
