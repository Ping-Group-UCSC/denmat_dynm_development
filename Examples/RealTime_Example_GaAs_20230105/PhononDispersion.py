#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from scipy.interpolate import interp1d
import os

#Read the phonon cell map and force matrix:
cellMap = np.loadtxt("totalE.phononCellMap")[:,0:3].astype(np.int)
if os.path.exists("totalE.phononOmegaSqCorr"):
  print("read totalE.phononOmegaSqCorr")
  forceMatrix = np.fromfile("totalE.phononOmegaSqCorr", dtype=np.float64)
else:
  print("read totalE.phononOmegaSq")
  forceMatrix = np.fromfile("totalE.phononOmegaSq", dtype=np.float64)
nCells = cellMap.shape[0]
nModes = int(np.sqrt(forceMatrix.shape[0] / nCells))
forceMatrix = np.reshape(forceMatrix, (nCells,nModes,nModes))

#Read the k-point path:
kpointsIn = np.loadtxt('bandstruct.kpoints', skiprows=2, usecols=(1,2,3))
nKin = kpointsIn.shape[0]
#--- Interpolate to a 10x finer k-point path:
nInterp = 1
xIn = np.arange(nKin)
x = (1./nInterp)*np.arange(1+nInterp*(nKin-1)) #same range with 10x density
kpoints = interp1d(xIn, kpointsIn, axis=0)(x)
nK = kpoints.shape[0]

#Calculate dispersion from force matrix:
#--- Fourier transform from real to k space:
forceMatrixTilde = np.tensordot(np.exp((2j*np.pi)*np.dot(kpoints,cellMap.T)), forceMatrix, axes=1)
#--- Diagonalize:
omegaSq, normalModes = np.linalg.eigh(forceMatrixTilde)
#--- Save:
omega = omegaSq.copy()
for q in range(omega.shape[0]):
  for m in range(omega.shape[1]):
    if omega[q,m] >= 0.:
      omega[q,m] = np.sqrt(omega[q,m])
    else:
      omega[q,m] = -np.sqrt(-omega[q,m])
np.savetxt("phfrq.dat", omega)
