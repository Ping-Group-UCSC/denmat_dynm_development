#!/usr/bin/env python
"""
Copyright 2018 Ravishankar Sundararaman

This file is part of JDFTx.

JDFTx is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

JDFTx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with JDFTx.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

#Atomic units:
eV = 1/27.21138505
Kelvin = 1./3.1577464e5
fs = 1./0.02418884326

#Read input file sigmaAC.in
T = 298*Kelvin
omegaMax = 10*eV
domega = 0.01*eV
polTheta = 0.
polPhi = 0.
for line in open('sigmaAC.in'):
	if line[0]=='#':
		continue #ignore comments
	tokens = line.split()
	if tokens[0]=='T':
		kT = float(tokens[1]) * Kelvin
		beta = 1/kT
	if tokens[0]=='omegaMax':
		omegaMax = float(tokens[1]) * eV
	if tokens[0]=='domega':
		domega = float(tokens[1]) * eV
	if tokens[0]=='polTheta':
		polTheta = float(tokens[1]) * np.pi/180
	if tokens[0]=='polPhi':
		polPhi = float(tokens[1]) * np.pi/180

#Read outputs of phononElectronLinewidth
#--- Gph.qList
qListDat = np.loadtxt("Gph.qList")
qpoints = qListDat[:,:3]
wq = qListDat[:,3] #weights
nq = wq.shape[0]
#--- Gph.dat
GphDat = np.loadtxt("Gph.dat")
omegaPh = np.reshape(GphDat[:,0], (nq,-1))
Gph = np.reshape(GphDat[:,2], (nq,-1)) #momentum-relaxing version
nModes = Gph.shape[1]
#--- Fermi level integrals from log file:
vv = np.zeros((3,3))
refLine = -10
iLine = 0
for line in open('phononElectronLinewidth.out'):
	if line.startswith('gEf = '):
		gEf = float(line.split()[2]) #in Eh^-1 per unit cell
	if line.startswith('Omega = '):
		Omega = float(line.split()[2]) #unit cell volume in a0^3
	#Read vv matrix:
	if line.startswith('vvEf:'):
		refLine = iLine
	if (iLine>refLine) and (iLine<=refLine+3):
		vv[iLine-refLine-1] = [ float(s) for s in line.split()[1:4] ]
	iLine += 1

#Frequency grid:
omega = np.arange(0., omegaMax, domega)

#Polarization direction:
sT = np.sin(polTheta); cT = np.cos(polTheta)
sP = np.sin(polPhi); cP = np.cos(polPhi)
pol = np.array([ sT*cP, sT*sP, cT ])

#Evaluate scattering rate for various frequencies:
def b(x):
	xReg = np.where(np.abs(x)<1e-6, 1e-6, x)
	return xReg/(1. - np.exp(-beta*xReg))
bDen = ( np.sum(np.sum(wq[None,:,None] * Gph[None,...] * b(omega[:,None,None]+omegaPh[None,...]), axis=-1), axis=-1)
	/ np.sum(np.sum(wq[None,:,None] * Gph[None,...], axis=-1), axis=-1) )
tauInv = (2*np.pi/(gEf*bDen)) * np.sum(np.sum(wq[None,:,None] * Gph[None,...] * b(omega[:,None,None]-omegaPh[None,...]), axis=-1), axis=-1)
#--- save data:
outDat = np.array([omega/eV, (1./tauInv)/fs]).T
np.savetxt('tauAC.dat', outDat, header='omega[eV] tauAC[fs]')

#Evaluate complex dielectric function for various frequencies:
omegaReg = np.maximum(omega,1e-6)
epsDrude = 1. - (4*np.pi) * (np.dot(pol, np.dot(vv, pol))/Omega) / (omegaReg*(omegaReg+1j*tauInv))
#--- save data:
outDat = np.array([omega/eV, np.real(epsDrude)]).T
np.savetxt('ReEpsDrude.dat', outDat, header='omega[eV] ReEpsDrude')
outDat = np.array([omega/eV, np.imag(epsDrude)]).T
np.savetxt('ImEpsDrude.dat', outDat, header='omega[eV] ImEpsDrude')

#Plot data:
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(omega/eV, (1./tauInv)/fs)
plt.xlabel(r'$\omega$ [eV]')
plt.ylabel(r'$\tau$ [fs]')

plt.figure(2)
plt.plot(omega/eV, np.real(epsDrude), label=r'Re$\epsilon$')
plt.plot(omega/eV, np.imag(epsDrude), label=r'Im$\epsilon$')
plt.xlabel(r'$\omega$ [eV]')
plt.ylabel(r'$\epsilon$')
plt.ylim([-10,10])
plt.legend()
plt.show()
