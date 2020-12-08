#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:25:28 2020

@author: ogurcan
"""
import os
import numpy as np
from ctypes import cdll,CFUNCTYPE,POINTER,c_double,c_int,byref

class mpcvode:
    def __init__(self, fn, y, dydt, t0, t1, **kwargs):
        self.libmpcvod = cdll.LoadLibrary(os.path.dirname(__file__)+'/libmpcvode.so')
        self.fnpytype=CFUNCTYPE(None, c_double, POINTER(c_double), POINTER(c_double))
        self.local_shape=y.shape
        self.global_shape=y.global_shape
        self.global_size = int(np.prod(y.global_shape)*
                               y.dtype.itemsize/np.dtype(float).itemsize)
        self.local_size = int(y.size*y.dtype.itemsize/np.dtype(float).itemsize)
        self.fn=fn
        self.kwargs=kwargs
        self.comm=y.comm
        self.t0=t0
        self.t1=t1
        self.y=y
        self.dydt=dydt
        self.t=t0
        self.state=0
        self.atol = kwargs.get('atol',1e-8)
        self.rtol = kwargs.get('rtol',1e-6)
        self.mxsteps = int(kwargs.get('mxsteps',10000))
        self.fnmpcvod=self.fnpytype(lambda x,y,z : self.fnforw(x,y,z))
        self.fn
        self.libmpcvod.init_solver(self.global_size,self.local_size,
                                   self.y.ctypes.data_as(POINTER(c_double)),
                                   self.dydt.ctypes.data_as(POINTER(c_double)),
                                   c_double(self.t0),self.fnmpcvod,c_double(self.atol),
                                   c_double(self.rtol),c_int(self.mxsteps));

    def fnforw(self,t,y,dydt):
        u=np.ctypeslib.as_array(y,(self.local_size,)).view(
            dtype=complex).reshape(self.local_shape)
        dudt=np.ctypeslib.as_array(dydt,(self.local_size,)).view(
            dtype=complex).reshape(self.local_shape)
        self.fn(t,u,dudt)

    def integrate_to(self,tnext):
        t=c_double()
        state=c_int()
        self.libmpcvod.integrate_to(c_double(tnext),byref(t),byref(state))
        self.t=t.value
        self.state=state.value

    def successful(self):
        return self.state==0
