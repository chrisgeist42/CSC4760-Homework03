using namespace std;
#include <iostream>
#include <assert.h>
#include <string>

#include <mpi.h>

#define GLOBAL_PRINT

class Domain
{
public:
  Domain(int _M, int _N, const char *_name="") : domain(new char[(_M+1)*(_N+1)]), M(_M), N(_N), name(_name)  {}
  virtual ~Domain() {delete[] domain;}
  char &operator()(int i, int j) {return domain[i*N+j];}
  char operator()(int i, int j)const {return domain[i*N+j];}

  int rows() const {return M;}
  int cols() const {return N;}

  const string & myname() const {return name;}

  char *rawptr() {return domain;}
  
protected:
  char *domain; 
  int M;
  int N;

  string name;
};

void zero_domain(Domain &domain);
void print_domain(Domain &domain, int rank);
void update_domain(Domain &new_domain, Domain &old_domain, int size, int myrank, MPI_Comm comm);
void parallel_code(int M, int N, int iterations, int size, int myrank, MPI_Comm comm);

int main(int argc, char **argv)
{
  int M, N;
  int iterations;

  if(argc < 4)
  {
    cout << "usage: " << argv[0] << " M N iterations" << endl;
    exit(0);
  }

  int size, myrank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  int array[3];
  if(myrank == 0)
  {
     M = atoi(argv[1]); N = atoi(argv[2]); iterations = atoi(argv[3]);

     array[0] = M;
     array[1] = N;
     array[2] = iterations;
     
  }
  MPI_Bcast(array, 3, MPI_INT, 0, MPI_COMM_WORLD);
  if(myrank != 0)
  {
    M = array[0];
    N = array[1];
    iterations = array[2];
  }

  parallel_code(M, N, iterations, size, myrank, MPI_COMM_WORLD);
  
  MPI_Finalize();
}

void parallel_code(int M, int N, int iterations, int size, int myrank, MPI_Comm comm)
{
  int nominal = M / size; int extra = M % size;
  int m = (myrank < extra) ? (nominal+1) : nominal;
  int n = N;
  
  Domain even_domain(m,n,"even Domain");
  Domain odd_domain(m,n,"odd Domain");

  zero_domain(even_domain);
  zero_domain(odd_domain);

#ifdef GLOBAL_PRINT
  Domain *global_domain = nullptr;
  int *thecounts = nullptr;
  int *displs = nullptr;

  if(0 == myrank)
  {
    thecounts  = new int[size];
    displs = new int[size];
    for(int i = 0; i < size; ++i)
      thecounts[i] = nominal*n; 
    for(int i = 0; i < extra; ++i)
      thecounts[i] += n;

    int count = displs[0] = 0;
    for(int i = 1; i < size; ++i)
    {
      count += thecounts[i-1];
      displs[i] = count;
    }
    
    global_domain = new Domain(M,N,"Global Domain");
    zero_domain(*global_domain);

    if((n >= 8) && (m >= 10))
    {
      (*global_domain)(0,(n-1)) = 1;
      (*global_domain)(0,0)     = 1;
      (*global_domain)(0,1)     = 1;

      (*global_domain)(8,5)     = 1;
      (*global_domain)(8,6)     = 1;
      (*global_domain)(8,7)     = 1;
      (*global_domain)(7,7)     = 1;
      (*global_domain)(6,6)     = 1;
    }
  }
  
  MPI_Scatterv(((myrank==0) ? global_domain->rawptr() : nullptr),
	       thecounts, displs, MPI_CHAR,
	       even_domain.rawptr(), m*n, MPI_CHAR, 0, comm);
  
#else  
  if((n >= 8) && (m >= 10))
  {
#if 0    
    even_domain(0,(n-1)) = 1;
    even_domain(0,0)     = 1;
    even_domain(0,1)     = 1;
    
    even_domain(3,5) = 1;
    even_domain(3,6) = 1;
    even_domain(3,7) = 1;

    even_domain(6,7) = 1;
    even_domain(7,7) = 1;
    even_domain(8,7) = 1;
    even_domain(9,7) = 1;
#else
    // blinker at top left, touching right...
    even_domain(0,(n-1)) = 1;
    even_domain(0,0)     = 1;
    even_domain(0,1)     = 1;

    // and a glider:
    even_domain(8,5)     = 1;
    even_domain(8,6)     = 1;
    even_domain(8,7)     = 1;
    even_domain(7,7)     = 1;
    even_domain(6,6)     = 1;
#endif    
  }
#endif  

#ifdef GLOBAL_PRINT
    if(0 == myrank)
    {  
      cout << "Initial State:" << endl;
      print_domain(*global_domain, myrank);
    }
#else
    cout << "Initial State:" << i << endl;
    print_domain(*even, myrank);
#endif  

  Domain *odd, *even;
  odd = &odd_domain;
  even = &even_domain;

  for(int i = 0; i < iterations; ++i)
  {
    update_domain(*odd, *even, size, myrank, comm);

#ifdef GLOBAL_PRINT
    if(0 == myrank)
      cout << "Iteration #" << i << endl;
    MPI_Gatherv(odd->rawptr(), m*n, MPI_CHAR,
		((myrank==0) ? global_domain->rawptr() : nullptr),
		thecounts, displs, MPI_CHAR, 0, comm);

    if(0 == myrank)
      print_domain(*global_domain, myrank);
#else
    cout << "Iteration #" << i << endl; print_domain(*odd, myrank);
#endif

    // swap pointers:
    Domain *temp = odd;
    odd  = even;
    even = temp;
  }
#ifdef GLOBAL_PRINT
  if(0 == myrank)
  {
    delete global_domain;
    delete[] thecounts;
    delete[] displs;
  }
#endif  
}

void zero_domain(Domain &domain)
{
  for(int i = 0; i < domain.rows(); ++i)
    for(int j = 0; j < domain.cols(); ++j)
      domain(i,j) = 0;
}

void print_domain(Domain &domain, int rank)
{
  cout << rank << ": " << domain.myname() << ":" <<endl;
  for(int i = 0; i < domain.rows(); ++i)
  {
    for(int j = 0; j < domain.cols(); ++j)
      cout << (domain(i,j) ? "*" : ".");
    cout << endl;
  }
}
      
void update_domain(Domain &new_domain, Domain &old_domain, int size, int myrank, MPI_Comm comm)
{
  MPI_Request request[4];
  
  int m = new_domain.rows();
  int n = new_domain.cols();
  char *top_row = new char[n];
  char *bottom_row = new char[n];

  char *top_halo = new char[n];
  char *bottom_halo = new char[n];

  const int top_row_index = 0; 
  const int bottom_row_index = m-1;

  const int TOP_HALO = 0, BOTTOM_HALO = 1;
  MPI_Irecv(top_halo, n, MPI_CHAR,    (myrank+size-1)%size, TOP_HALO, comm, &request[2]);
  MPI_Irecv(bottom_halo, n, MPI_CHAR, (myrank+1)%size, BOTTOM_HALO, comm, &request[3]);

  for(int j = 0; j < n; ++j)
  {
      top_row[j] = old_domain(top_row_index,j);
  }
  MPI_Isend(top_row, n, MPI_CHAR, (myrank-1+size)%size, BOTTOM_HALO, comm, &request[0]);

  for(int j = 0; j < n; ++j) // fill in the bottom row
  {
     bottom_row[j] = old_domain(bottom_row_index,j);
  }
  MPI_Isend(bottom_row, n, MPI_CHAR, (myrank+1)%size, TOP_HALO, comm, &request[1]);

  MPI_Waitall(4, request, MPI_STATUSES_IGNORE);
  for(int j = 0; j < n; ++j)
  {
      int neighbor_count = 0;

      for(int delta_i = 0; delta_i <= 1; ++delta_i)
      {
	for(int delta_j = -1; delta_j <= 1; ++delta_j)
	{
	  if(delta_i == 0 && delta_j == 0) //skip self
	    continue;

	  if(old_domain((top_row_index+delta_i),
			(j+delta_j+old_domain.cols())%old_domain.cols()))
	     ++neighbor_count;
	}
      }
      for(int delta_j = -1; delta_j <= 1; ++delta_j)
      {
        if(top_halo[(j+delta_j+old_domain.cols())%old_domain.cols()])
	  ++neighbor_count;
      }
      
      new_domain(top_row_index,j)
	= update_the_cell(old_domain(top_row_index,j),neighbor_count);
  }
  for(int j = 0; j < new_domain.cols(); ++j)
  {
      int neighbor_count = 0;

      for(int delta_i = -1; delta_i <= 0; ++delta_i)
      {
	for(int delta_j = -1; delta_j <= 1; ++delta_j)
	{
	  if(delta_i == 0 && delta_j == 0) //skip self
	    continue;

	  if(old_domain((bottom_row_index+delta_i),
			(j+delta_j+old_domain.cols())%old_domain.cols()))
	     ++neighbor_count;
	}
      }
      for(int delta_j = -1; delta_j <= 1; ++delta_j)
      {
        if(bottom_halo[(j+delta_j+old_domain.cols())%old_domain.cols()])
	  ++neighbor_count;
      }
      
      new_domain(bottom_row_index,j)
	= update_the_cell(old_domain(bottom_row_index,j), neighbor_count);
  }
  for(int i = (top_row_index+1); i < bottom_row_index ; ++i)
  {
    for(int j = 0; j < n; ++j)
    {
      int neighbor_count = 0;
      for(int delta_i = -1; delta_i <= 1; ++delta_i)
      {
	for(int delta_j = -1; delta_j <= 1; ++delta_j)
	{
	  if(delta_i == 0 && delta_j == 0)
	    continue;

	  if(old_domain((i+delta_i+old_domain.rows())%old_domain.rows(),
			(j+delta_j+old_domain.cols())%old_domain.cols()))
	     ++neighbor_count;
	}
      }
      new_domain(i,j) = update_the_cell(old_domain(i,j), neighbor_count);
    }
  }
  delete[] bottom_halo;
  delete[] top_halo;
  delete[] bottom_row;
  delete[] top_row;
}
