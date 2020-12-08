#include <nvector/nvector_parallel.h>
#include <mpi.h>

typedef struct mpcv_pars{
  double *y, *dydt;
  double t0,t;
  int N,Nloc;
  MPI_Comm comm;
  void *solver;
  N_Vector uv;
  void (*fnpy)(double, double *, double *);
}mpcv_pars;

void init_solver(int N,int Nloc, double *y, double *dydt, double t0,
		 void (*fnpy)(double,double *,double *),
		 double atol, double rtol, int mxsteps);
void integrate_to(double tnext, double *t, int *state);
