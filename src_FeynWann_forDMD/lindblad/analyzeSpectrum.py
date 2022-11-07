#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys

if len(sys.argv)<2:
	print('Usage: analyzeEigs.py <lindblad-spectrum-log-file> [<tMax_ps>=auto]')
	exit(1)

fname = sys.argv[1]
tMax_ps = sys.argv[2] if (len(sys.argv)>=3) else 'auto'

eigRead = False
eigData = []
for line in open(fname):
	if line=='\n':
		eigRead = False
	if eigRead:
		eigData.append([ float(s) for s in line.split() ])
	if line.find("Re(eig)") >= 0:
		eigRead = True
eigData = np.array(eigData)

ps = 1000./0.02418884
E = (eigData[:,0] + 1j*eigData[:,1])*ps #convert to ps^-1
Spert = eigData[:,2:5]
Smat = eigData[:,5:8]

#Determine relevant timescales:
tauInv = -E.real
omega = E.imag
tauInvMin = tauInv[tauInv>1e-10].min() #min tauInv other than the constant mode
if tMax_ps=='auto' or tMax_ps=='Auto':
	tMax_ps = 3./tauInvMin
else:
	tMax_ps = float(tMax_ps)

#Split real and complex eigenvalues:
selRe = np.where(E.imag==0.)[0] #real components
selCp = np.where(E.imag>0.)[0] #conjugate pairs: positive Im
selCm = np.where(E.imag<0.)[0] #conjugate pairs: negative Im
#--- real components:
Ere = E[selRe].real
SpertRe = Spert[selRe]
SmatRe = Smat[selRe]
#--- complex components
Ec = E[selCp]
SpertC = Spert[selCp] + 1j*Spert[selCm]
SmatC = Smat[selCp] + 1j*Smat[selCm]
#--- make original arrays complex
Spert = Spert + 0j; Spert[selCp] = SpertC; Spert[selCm] = SpertC.conj()
Smat = Smat + 0j; Smat[selCp] = SmatC; Smat[selCm] = SmatC.conj()

#Plot overall collection of eigenvalues:
plt.figure(1)
eigWeightVec = np.abs(Spert * Smat) #weight of each mode in spin dynamics of each direction
eigWeightVec *= 1./eigWeightVec.max(axis=0)[None,:] #normalize max weight in each direction to 1
eigWeight = np.max(eigWeightVec, axis=1) #max weight of any direction
colorCoords = eigWeightVec*(eigWeightVec.max(axis=1)/eigWeightVec.sum(axis=1))[:,None] #barycentric coordinates for 3D color mapping
colorNone = np.array([[0.9,0.9,0.9]]) #color for spin-irrelevant modes
colorXYZ = np.array([
	[.3,.3,1.],   #color for an Sx-relevant mode
	[0.,.8,0.],   #color for an Sy-relevant mode
	[1.,0.,0.] ]) #color for an Sz-relevant mode
colors = colorNone + (colorCoords[:,:,None]*(colorXYZ-colorNone)[None,:,:]).sum(axis=1)
plt.scatter(tauInv, omega, c=colors, marker='+')
plt.xscale('log')
plt.yscale('symlog', linthresh=tauInvMin)
plt.xlim(0.5*tauInvMin, None) #capture smallest modes, but avoid constant one
plt.ylim(E.imag.min(), E.imag.max())
plt.xlabel(r'-Re(eig) = $\tau^{-1}$ [ps$^{-1}$]')
plt.ylabel(r'Im(eig) = $\omega$ [ps$^{-1}$]')
plt.axhline(0, color='k', ls='dotted', lw=1)
#--- invisible points just to compose legend:
plt.scatter(-1,0, color=colorNone[0], marker='+', label='Irrelevant')
plt.scatter(-1,0, color=colorXYZ[0], marker='+', label=r'$S_x$-relevant')
plt.scatter(-1,0, color=colorXYZ[1], marker='+', label=r'$S_y$-relevant')
plt.scatter(-1,0, color=colorXYZ[2], marker='+', label=r'$S_z$-relevant')
plt.legend()

#Print most spin-relevant eigenvalues:
print('Spin-relevant real modes:')
nRelevant = len(np.where(eigWeight[selRe] > 0.1)[0])
iSorted = eigWeight[selRe].argsort()[::-1][:nRelevant] #descending order
for i in iSorted:
	Sweight = eigWeightVec[selRe[i]]
	print("tau[ps]: {:9.3f}  SpinRelevance:  {:.3f} {:.3f} {:.3f}".format(-1./Ere[i], *Sweight))
print('Spin-relevant complex mode pairs (printing one per pair):')
nRelevant = len(np.where(eigWeight[selCp] > 0.1)[0])
iSorted = eigWeight[selCp].argsort()[::-1][:nRelevant] #descending order
for i in iSorted:
	Sweight = eigWeightVec[selCp[i]]
	print("tau[ps]: {:9.3f}  SpinRelevance:  {:.3f} {:.3f} {:.3f}  Period[ps]: {:9.3f}".format(
		-1./Ec[i].real, Sweight[0], Sweight[1], Sweight[2], (2*np.pi)/Ec[i].imag))

#Time evolution (S_a evolution under B_a perturbation for each component a):
t = np.linspace(0., tMax_ps, 1000)
S = np.dot(np.exp(Ere[None,:]*t[:,None]), SmatRe * SpertRe) #real part
S += np.dot(np.exp(Ec[None,:]*t[:,None]), SmatC * SpertC.conj()).real #Re() accounts for contribution from c.c. pairs
#--- plot
plt.figure(2)
for i,iName in enumerate(['x','y','z']):
	plt.plot(t, S[:,i], label=iName, color=colorXYZ[i])
plt.axhline(0, color='k', ls='dotted', lw=1)
plt.legend()
plt.xlabel('t [ps]')
plt.ylabel('Spin')
plt.ticklabel_format(axis="both", style="sci", scilimits=(-3,3))
plt.show()
