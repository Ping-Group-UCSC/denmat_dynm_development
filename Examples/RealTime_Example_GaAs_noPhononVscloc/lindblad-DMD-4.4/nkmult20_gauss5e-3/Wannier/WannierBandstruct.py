#!/usr/bin/env python
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import re
import sys
import os

#Optional argument fo spin-polarized mode:
suffix = ""
nSpins = 1
iSpin = 0
if len(sys.argv)>1:
	suffix = sys.argv[1]
	nSpins = 2
	if suffix=="Up":
		iSpin = 0
	elif suffix=="Dn":
		iSpin = 1
	else:
		print("Argument must be 'Up' or 'Dn' and only specified for spin-polarized calculations.")
		exit(1)

np.set_printoptions(linewidth=250, precision=5)
nInterp = 1

#Read the band structure k-points:
kpointsIn = np.loadtxt('bandstruct.kpoints', skiprows=2, usecols=(1,2,3))
nKin = kpointsIn.shape[0]
#--- Interpolate to a finer k-point path:
xIn = np.arange(nKin)
x = (1./nInterp)*np.arange(1+nInterp*(nKin-1)) #same range with 10x density
kpoints = interp1d(xIn, kpointsIn, axis=0)(x)
nK = kpoints.shape[0]

#Read DFT band structure:
Edft = np.fromfile('bandstruct.eigenvals').reshape(nSpins,nKin,-1)[iSpin]

#Get k-point folding and mu/VBM from totalE.out:
mu = np.nan
initDone = False
for line in open('totalE.out'):
	if line.startswith('Initialization completed'):
		initDone = True
	if initDone and line.find('FillingsUpdate:')>=0:
		mu = float(line.split()[2])
	if (not initDone) and line.startswith('kpoint-folding'):
		kfold = np.array([int(tok) for tok in line.split()[1:4]])
	if (not initDone) and line.startswith('nElectrons:'):
		nElectrons = float(line.split()[1])
		nValence = int(nElectrons/2) #number of valence bands
		mu = np.max(Edft[:,:nValence]) #VBM
kfoldProd = np.prod(kfold)
kStride = np.array([kfold[1]*kfold[2], kfold[2], 1])

#Read the wannier cell map and weightsHamiltonian:    
cellMapIn = np.loadtxt("wannier.mlwfCellMap"+suffix)
cellMap = cellMapIn[:,:3].astype(np.int)
cellMapCart = cellMapIn[:,3:]
Wwannier = np.fromfile("wannier.mlwfCellWeights"+suffix, dtype=np.float64)
nCells = cellMap.shape[0]
nBands = int(np.sqrt(Wwannier.shape[0] / nCells))
Wwannier = Wwannier.reshape((nCells,nBands,nBands)).swapaxes(1,2)

#Read and expand Wannier Hamiltonian:
Hreduced = np.fromfile("wannier.mlwfH"+suffix, dtype=np.complex128).reshape((kfoldProd,nBands,nBands)).swapaxes(1,2)
iReduced = np.dot(np.mod(cellMap, kfold[None,:]), kStride)
Hwannier = Wwannier * Hreduced[iReduced]

#Calculate band structure from MLWF Hamiltonian:
#--- Fourier transform from MLWF to k space:
Hk = np.tensordot(np.exp((2j*np.pi)*np.dot(kpoints,cellMap.T)), Hwannier, axes=1)
#--- Diagonalize:
Ek,_ = np.linalg.eigh(Hk)
#--- Save:
np.savetxt("wannier.eigenvals", Ek)

'''
#Plot:
iKdft = np.arange(0, nK, nInterp)
plt.plot(iKdft, Edft, 'k')
plt.plot(Ek, 'r')
plt.xlim(0, nK-1)
plt.ylabel("E [Eh]")
#--- read and add plot labels:
for line in open('bandstruct.plot'):
	if line.startswith('set xtics'):
		tokens = re.split('[ ,"]*', line)
		xticLabels = [ (r'$\Gamma$' if token=='Gamma' else token) for token in tokens[3:-1:2] ]
		xticPos = [ int(token)*nInterp for token in tokens[4:-1:2] ]
		plt.xticks(xticPos, xticLabels)
#--- annotate mu or VBM as applicable:
if not np.isnan(mu):
	plt.axhline(mu, linestyle='dashed', color='black', linewidth=1)
#--- read and annotate Wannier windows:
wannierInFile = (('wannier.in'+suffix) if os.path.exists('wannier.in'+suffix) else 'wannier.in')
for line in open(wannierInFile):
	key = 'innerWindow'
	iKey = line.find(key)
	if iKey>=0:
		tokens = line[iKey+len(key):].split()
		iwMin,iwMax = [ float(token) for token in tokens[:2] ]
		plt.axhline(iwMin, linestyle='dotted', color='red', linewidth=1)
		plt.axhline(iwMax, linestyle='dotted', color='red', linewidth=1)
	key = 'outerWindow'
	iKey = line.find(key)
	if iKey>=0:
		tokens = line[iKey+len(key):].split()
		owMin,owMax = [ float(token) for token in tokens[:2] ]
		plt.axhline(owMin, linestyle='dotted', color='red', linewidth=1)
		plt.axhline(owMax, linestyle='dotted', color='red', linewidth=1)
		plt.ylim(owMin, owMax)
plt.show()
'''
