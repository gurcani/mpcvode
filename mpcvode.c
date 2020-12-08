#include <stdlib.h>
#include <stdio.h>
#include <cvode/cvode.h>                          /* prototypes for CVODE fcts., consts.          */
#include <nvector/nvector_parallel.h>             /* access to MPI-parallel N_Vector              */
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h> /* access to the fixed point SUNNonlinearSolver */
#include <sundials/sundials_types.h>              /* definition of type realtype                  */
#include <mpi.h> /* MPI constants and types */
#include "mpcvode.h"
//#include <omp.h>

mpcv_pars *p_glob;

static int fnmpcvode(realtype t, N_Vector y, N_Vector dydt, void *fdata){
  mpcv_pars *p=(mpcv_pars*)fdata;
  p->fnpy(t,NV_DATA_P(y),NV_DATA_P(dydt));
  return 0;
}

void init_solver(int N,int Nloc, double *y, double *dydt, double t0, void (*fnpy)(double,double *,double *), double atol, double rtol, int mxsteps){
  SUNNonlinearSolver NLS;
  int state;
  mpcv_pars *p;
  p=malloc(sizeof(mpcv_pars));
  p->N=N;
  p->Nloc=Nloc;
  p->comm=MPI_COMM_WORLD;
  p->y=y;
  p->t0=t0;
  p->dydt=dydt;
  p->fnpy=fnpy;
  p->uv=N_VMake_Parallel(p->comm,Nloc,N,y);
  p->solver=CVodeCreate(CV_ADAMS);
  state = CVodeSetUserData(p->solver, p);
  state = CVodeSetMaxNumSteps(p->solver, mxsteps);
  state = CVodeInit(p->solver, fnmpcvode,t0,p->uv);
  state = CVodeSStolerances(p->solver, rtol, atol);
  NLS = SUNNonlinSol_FixedPoint(p->uv, 0);
  state = CVodeSetNonlinearSolver(p->solver, NLS);
  p_glob=p;
};

void integrate_to(double tnext, double *t, int *state){
  mpcv_pars *p=p_glob;
  *state=CVode(p->solver, tnext, p->uv, &(p->t), CV_NORMAL);
  *t=p->t;
}
