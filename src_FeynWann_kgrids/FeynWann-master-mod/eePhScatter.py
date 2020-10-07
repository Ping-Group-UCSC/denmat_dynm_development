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
from scipy.special import spherical_in

#Atomic units:
eV = 1/27.21138505
Kelvin = 1./3.1577464e5
fs = 1./0.02418884326

#Read outputs of phononElectronLinewidth
#--- Gph.qList
qListDat = np.loadtxt("Gph.qList")
qpoints = qListDat[:,:3]
wq = qListDat[:,3] #weights
nq = wq.shape[0]
#--- Gph.dat
GphDat = np.loadtxt("Gph.dat")
omegaPh = np.reshape(GphDat[:,0], (nq,-1))
Gph = np.reshape(GphDat[:,1], (nq,-1))
nModes = Gph.shape[1]

#Read electronic density of states:
eDOSdat = np.loadtxt("eDOS.dat")
eDOS_E = eDOSdat[:,0]*eV #convert to Eh
eDOS_g = eDOSdat[:,1]/eV #convert to Eh^-1
gEf = np.interp(0., eDOS_E, eDOS_g) #density of states at Fermi level in Eh^-1/unit cell

#Helper function for evaluating the frequency integrals:
def gamma(x, eta):
	#sinh(x) / x:
	def sinch(x):
		xMax = 20.
		return spherical_in(0, np.minimum(xMax, np.abs(x))) #modified spherical bessel
	gammaS = np.pi/((x*eta) * (sinch(0.5*x)**2)) #singular part
	gamma0 = (
		(-1.+x*(0.10124873+x*(-0.00342582+x*0.01213954)))
		/ (1.+x*(-0.67478118+x*(0.30375048+x*(-0.05723733+x*0.00500147))))
		/ x )
	gamma1 = (
		(0.+x*(0.27074268+x*(-0.02150408+x*-0.03999233)))
		/ (1.+x*(-0.47001105+x*(0.15998854+x*(-0.02279949+x*0.0016412))))
		* np.exp(-0.5*x) )
	return gammaS + gamma0 + eta*gamma1

#Evaluate scattering rate for various temperatures:
T = np.logspace(0, 3, 61)*Kelvin #1-1000 K
beta = 1./T
x = omegaPh[None,...] * beta[:,None,None]
GppArr = [ 0., 0.001, 0.003, 0.01 ] #Various values for competing phonon-phonon scattering loss tangent
tauEPEinv = []
for Gpp in GppArr:
	eta = np.pi * (Gpp + Gph[None,...]) #phonon loss tangent = (1/2)tau^-1/omegaPh = pi G
	integral = np.sum(np.sum(wq[None,:,None] * (Gph[None,...]**2) * gamma(x, eta), axis=-1), axis=-1)
	tauEPEinv.append((2*np.pi/gEf) * integral)
tauEPEinv = np.array(tauEPEinv).T

#Save data:
outDat = np.hstack([T[:,None]/Kelvin, (1./tauEPEinv)/fs])
np.savetxt('tauEEph.dat', outDat, header='T[K] tauEEph[fs]')

#Plot data:
import matplotlib.pyplot as plt
for iGpp,Gpp in enumerate(GppArr):
	plt.loglog(T/Kelvin, (1./tauEPEinv[:,iGpp])/fs, label='$G_{\mathrm{pp}}='+str(Gpp)+'$')
plt.xlabel(r'$T$ [K]')
plt.ylabel(r'$\tau_{\mathrm{epe}}$ [fs]')
plt.legend()
plt.show()
