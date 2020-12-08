from mpi4py import MPI
import numpy as np
from mpcvode import mpcvode
import matplotlib.pylab as plt

def splitmpi(shape,rank,size,axis=-1,Nsp=0):
    sp=list(shape)
    if(Nsp==0):
        Nsp=sp[axis]
    nperpe = int(Nsp/size)
    nrem = Nsp - size*nperpe
    n = nperpe+(rank < nrem)
    start = rank*nperpe+min(rank,nrem)
    off=np.zeros(len(sp),dtype=int)
    sp[axis]=n
    off[axis]=start
    return sp,off

class distarray(np.ndarray):
    def __new__(self,shape,dtype=float,buffer=None,
                offset=0,strides=None,order=None,
                axis=-1,Nsp=0,comm=MPI.COMM_WORLD):
        dims=len(shape)
        locshape,loc0=splitmpi(shape, comm.rank, comm.size, axis, Nsp)
        if(buffer==None):
            buffer=np.zeros(locshape,dtype)
        else:
            if(dtype!=buffer.dtype):
                print("dtype!=buffer.dtype, ignoring dtype argument")
        dtype=buffer.dtype
        obj=super(distarray, self).__new__(self,locshape,dtype,buffer,offset,strides,order)
        obj.loc0=loc0
        obj.global_shape=shape
        obj.local_slice = tuple([slice(loc0[l],loc0[l]+locshape[l],None) for l in range(dims)])
        obj.comm=comm
        return obj
    
phi=distarray((4,4),dtype=complex,comm=MPI.COMM_WORLD)
dphidt=distarray((4,4),dtype=complex,comm=MPI.COMM_WORLD)
gam=0.1
phi[:,:]=[ [i+j for j in np.r_[phi.local_slice[1]] ] 
          for i in np.r_[phi.local_slice[0]] ]
phi0=phi.copy()
def fntest(t,y,dydt):
    dydt[:,:] = gam*y[:,:]

mpcv=mpcvode(fntest,phi,dphidt,0.0,100.0,atol=1e-12,rtol=1e-8)
t=np.arange(10.0)
z=np.zeros(10,dtype=complex)
z[0]=phi[0,0]
for l in range(1,t.shape[0]):
    mpcv.integrate_to(t[l])
    z[l]=mpcv.y[0,0]
plt.plot(t,z.real,'x',t,z[0].real*np.exp(0.1*t),'--')
plt.legend(['numerical solution',str(z[0].real)+'*exp(0.1*t)'])
plt.show()