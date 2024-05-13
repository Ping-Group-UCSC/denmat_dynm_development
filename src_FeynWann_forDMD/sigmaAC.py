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

import matplotlib.pyplot as plt
import numpy as np

# Atomic units:
eV = 1/27.21138505
Kelvin = 1./3.1577464e5
fs = 1./0.02418884326

# Read input file sigmaAC.in
params = {
    tokens[0]: tokens[1]
    for line in open('sigmaAC.in')
    if (line[0] != '#') and (len(tokens := line.split()) >= 2)
}
kT = float(params.get("T", 298)) * Kelvin
beta = 1.0 / kT
omegaMax = float(params.get("omegaMax", 10)) * eV
domega = float(params.get("domega", 0.01)) * eV
polTheta = np.deg2rad(float(params.get("polTheta", 0.)))
polPhi = np.deg2rad(float(params.get("polPhi", 0.)))

# Read outputs of phononElectronLinewidth
# --- Gph.qList
qListDat = np.loadtxt("Gph.qList")
qpoints = qListDat[:, :3]
wq = qListDat[:, 3]  # weights
nq = wq.shape[0]
# --- Gph.dat
GphDat = np.loadtxt("Gph.dat")
omegaPh = np.reshape(GphDat[:, 0], (nq, -1))
Gph = np.reshape(GphDat[:, 2], (nq, -1))  # momentum-relaxing version
nModes = Gph.shape[1]
# --- Fermi level integrals from log file:
vv = np.zeros((3, 3))
refLine = -10
for iLine, line in enumerate(open('phononElectronLinewidth.out')):
    if line.startswith('gEf = '):
        gEf = float(line.split()[2])  # in Eh^-1 per unit cell
    if line.startswith('Omega = '):
        Omega = float(line.split()[2])  # unit cell volume in a0^3
    # Read vv matrix:
    if line.startswith('vvEf:'):
        refLine = iLine
    if refLine < iLine <= (refLine + 3):
        vv[iLine-refLine-1] = [float(s) for s in line.split()[1:4]]

# Frequency grid:
omega = np.arange(0., omegaMax, domega)

# Polarization direction:
sT, cT = np.sin(polTheta), np.cos(polTheta)
sP, cP = np.sin(polPhi), np.cos(polPhi)
pol = np.array([sT * cP, sT * sP, cT])


def b(x):
    beta_x = np.copysign(np.maximum(beta * abs(x), 1E-6), x)
    return -beta_x / np.expm1(-beta_x)


# Evaluate scattering rate for various frequencies:
pm = np.array([+1, -1])
nPh = 1.0 / np.expm1(beta * np.maximum(omegaPh, 1E-6))
nPh_mp = np.array([nPh, nPh + 1.0])
b_pm = b(
    omega[:, None, None, None] + pm[None, :, None, None] * omegaPh[None, None]
)
tauInv = (
    np.einsum('q, qa, osqa, sqa -> o', wq, Gph, b_pm, nPh_mp)
    / (gEf * b(omega))
)
# --- save data:
outDat = np.array([omega/eV, (1./tauInv)/fs]).T
np.savetxt('tauAC.dat', outDat, header='omega[eV] tauAC[fs]')

# Evaluate complex dielectric function for various frequencies:
omegaReg = np.maximum(omega, 1e-6)
epsDrude = 1. - (4*np.pi) * (pol @ vv @ pol) / (
    Omega * omegaReg * (omegaReg + 1j*tauInv)
)
# --- save data:
outDat = np.array([omega/eV, np.real(epsDrude)]).T
np.savetxt('ReEpsDrude.dat', outDat, header='omega[eV] ReEpsDrude')
outDat = np.array([omega/eV, np.imag(epsDrude)]).T
np.savetxt('ImEpsDrude.dat', outDat, header='omega[eV] ImEpsDrude')

# Plot data:
plt.figure(1)
plt.plot(omega/eV, (1./tauInv)/fs)
plt.xlabel(r'$\omega$ [eV]')
plt.ylabel(r'$\tau$ [fs]')
plt.yscale('log')
plt.xscale('log')

plt.figure(2)
plt.plot(omega/eV, np.real(epsDrude), label=r'Re$\epsilon$')
plt.plot(omega/eV, np.imag(epsDrude), label=r'Im$\epsilon$')
plt.xlabel(r'$\omega$ [eV]')
plt.ylabel(r'$\epsilon$')
plt.ylim([-10, 10])
plt.legend()
plt.show()
