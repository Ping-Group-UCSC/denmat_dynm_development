#!/usr/bin/python
from __future__ import print_function
#import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import re

np.set_printoptions(linewidth=250, precision=5)
nInterp = 1

#Read lattice vectors:
R = np.zeros((3,3))
iLine = 0
refLine = -10
for line in open('totalE.out'):
	if line.find('Initializing the Grid') >= 0:
		refLine = iLine
	rowNum = iLine - (refLine+2)
	if rowNum>=0 and rowNum<3:
		R[rowNum,:] = [ float(x) for x in line.split()[1:-1] ]
	if rowNum==3:
		break
	iLine += 1

cellVol = np.linalg.det(R)
G = (2*np.pi)*np.linalg.inv(R)

#Read the phonon cell map and Hamiltonian:    
cellMapIn = np.loadtxt("totalE.phononCellMap")
cellMap = cellMapIn[:,:3].astype(np.int)
cellMapCart = cellMapIn[:,3:]
Hphonon = np.fromfile("totalE.phononOmegaSq", dtype=np.float64)
nCells = cellMap.shape[0]
nModes = int(np.sqrt(Hphonon.shape[0] / nCells))
nAtoms = nModes//3
Hphonon = np.reshape(Hphonon, (nCells,nModes,nModes))
cellWeights = np.reshape(np.fromfile("totalE.phononCellWeights"), (nCells,nAtoms,nAtoms))
Nsup = np.mean(np.sum(cellWeights, axis=0))
print('Number of supercells:', Nsup)
#Read the band structure k-points:
kpointsIn = np.loadtxt('bandstruct3.kpoints', skiprows=2, usecols=(1,2,3))
nKin = kpointsIn.shape[0]
#--- Interpolate to a finer k-point path:
xIn = np.arange(nKin)
x = (1./nInterp)*np.arange(1+nInterp*(nKin-1)) #same range with 10x density
kpoints = interp1d(xIn, kpointsIn, axis=0)(x)
nK = kpoints.shape[0]
#--- regularize Gamma point:
for ik in range(nK):
	if np.linalg.norm(kpoints[ik]) < 1e-6:
		ikNext = (ik+1 if ik+1<nK else ik-1)
		kpoints[ik] = kpoints[ikNext] * 1e-4/np.linalg.norm(kpoints[ikNext])

#Calculate band structure from MLWF Hamiltonian:
#--- Fourier transform from MLWF to k space:
Hk = np.tensordot(np.exp((2j*np.pi)*np.dot(kpoints,cellMap.T)), Hphonon, axes=1)
#--- Diagonalize:
Ek,_ = np.linalg.eigh(Hk)

#LOTO correction
#Get phonon basis at atom masses:
phononBasis = np.loadtxt('totalE.phononBasis', usecols=[2,3,4])
invSqrtM = np.sqrt(np.sum(phononBasis**2, axis=1)) #in atomic units
sqrtM = 1./invSqrtM

#Read dielectric and Zeff tensors
epsInf = np.loadtxt("totalE.epsInf")
effZ = (np.loadtxt("totalE.Zeff")).flatten()
effZ = np.reshape(effZ, (-1, 3,3)) #natoms, gamma and beta directions
#-- Test sum rule:
effZ -= np.mean(effZ, axis=0)[None,...]
print('effZ after sum rule:\n', effZ)

#Calculate force correction:
corrEk = np.zeros((kpoints.shape[0], nModes))
HkCurAll = np.zeros((kpoints.shape[0], nModes, nModes), dtype=np.complex128)
for ik,k in enumerate(kpoints):
	kCart = np.dot(k, G)
	qdotz = np.dot(effZ, kCart) #nAtoms x 3
	num = np.einsum('Rij,ia,jb->Riajb', cellWeights, qdotz, qdotz)
	den = np.dot(kCart, np.dot(epsInf, kCart))
	Fcorr = (4*np.pi/(Nsup*cellVol * den)) * num
	HphononCorr = np.einsum('Rxy,x,y->Rxy', Fcorr.reshape((nCells,nModes,nModes)), invSqrtM, invSqrtM)
	HkCur = np.tensordot(np.exp((2j*np.pi)*np.dot(cellMap,k)), Hphonon + HphononCorr, axes=1)
	HkCurAll[ik, :, :] = HkCur
	corrEk[ik],_ = np.linalg.eigh(HkCur)

#Plot:
unit = 1e-3/27.2114; unitName='meV'
#unit = 0.000151983; unitName='THz'
#omega = np.copysign(np.sqrt(np.abs(Ek)), Ek)
corrOmega = np.copysign(np.sqrt(np.abs(corrEk)), corrEk)
np.savetxt("phfrq_loto.dat", corrOmega)
'''
plt.plot(omega/unit, 'k-')
plt.plot(corrOmega/unit, 'r')
#plt.xlim(0, omega.shape[0])
plt.ylim(np.min(omega), None)
plt.ylabel("omega ["+unitName+"]")
#--- read and add plot labels:
for line in open('bandstruct.plot'):
	if line.startswith('set xtics'):
		tokens = re.split('[ ,"]*', line)
		xticLabels = [ (r'$\Gamma$' if token=='Gamma' else token) for token in tokens[3:-1:2] ]
		xticPos = np.array([ int(token)*nInterp for token in tokens[4:-1:2] ])
		plt.xticks(xticPos, xticLabels)
		xticMid = 0.5*(xticPos[1:]+xticPos[:-1])
		for x in np.concatenate((xticPos,xticMid)):
			plt.axvline(x, linestyle='dotted', color='black')
		plt.axhline(0, linestyle='dotted', color='black')
plt.plot([], c = 'r', label="LOTO")
plt.legend()
#plt.savefig("phononBand.png")
plt.show()
'''
