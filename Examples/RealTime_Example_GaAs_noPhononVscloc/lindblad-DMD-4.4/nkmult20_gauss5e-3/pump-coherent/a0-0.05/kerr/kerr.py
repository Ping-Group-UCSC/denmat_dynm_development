#!/usr/bin/env python
import numpy as np
import os.path
import subprocess
import sys
sys.path.insert(1, '/export/data/share/jxu/jdftx_codes/tools')
from kramersKronig import KK

d = 100 *10/0.529177 # thickness for Faraday rotation, 10/0.529177 is nm to au
prefac_fara = d/2/137 # d/2/c

# Based on based on PRB 53, 296 (1996)

def optical(Eps):
  n = np.sqrt(Eps)
  r = np.true_divide(1 - n, 1 + n)
  r[abs(r) < 1e-30] = 1e-30
  R = abs(r)**2
  R[abs(R) < 1e-40] = 1e-40
  #t = 1./(1+0.5*sig)
  #T = abs(t)**2
  #A = 1 - T - R
  return n, r, R # avoid zero in divide

step = 1 # energy step in ../probe_results/imEps.0 / energy step in ImEpsDirectLC.dat

# read ImEps at ground state over a wide photon energy range
# energy step in ImEpsDirectXX.dat must be the same as that in imEps.*
e = np.loadtxt("ImEpsDirectLC.dat", usecols=(0,))
imEpsGS = np.loadtxt("ImEpsDirectLC.dat", usecols=(1,))

# read energy grids of probe
# imEps.* must be in directory data
e_probe = np.loadtxt("../probe_results/imEps.0.dat", usecols=(0,))[::step]
ne_probe = e_probe.shape[0]
ie = np.argwhere(np.abs(e - e_probe[0]) < 1e-6)[0,0]

os.system("rm -r kerr_results")
os.mkdir("kerr_results")
dirk = "kerr_results/"
np.savetxt(dirk+"omega.dat", e)

it = 0
fim = "../probe_results/imEps." + str(it) + ".dat"
epsGS = KK(e, imEpsGS) + 1j*imEpsGS
nGs, rGS, RGS = optical(epsGS)

while os.path.isfile(fim): # loop on time
	
  # ImEps(t) = \Delta ImEps(t) + ImEps_{Ground-State}
  imeps_lc = np.copy(imEpsGS)
  imeps_lc[ie:ie+ne_probe] = imEpsGS[ie:ie+ne_probe] + np.loadtxt(fim, usecols=(1,))[::step]
  imeps_rc = np.copy(imEpsGS)
  imeps_rc[ie:ie+ne_probe] = imEpsGS[ie:ie+ne_probe] + np.loadtxt(fim, usecols=(2,))[::step]

  # compute reflectivity
  eps_lc = KK(e, imeps_lc) + 1j*imeps_lc
  n_lc, r_lc, R_lc = optical(eps_lc)
  eps_rc = KK(e, imeps_rc) + 1j*imeps_rc
  n_rc, r_rc, R_rc = optical(eps_rc)

  # compute Kerr rotation
  kerr = 1j * np.true_divide(r_lc - r_rc, r_lc + r_rc)
  fara = prefac_fara * e * np.real(n_lc - n_rc)
  dR_lc = R_lc - RGS
  dR_rc = R_rc - RGS
  if it == 0:
    kerrR = np.zeros(dR_lc.shape[0])
  else:
    kerrR = np.true_divide(dR_lc - dR_rc, dR_lc + dR_rc)
  np.savetxt(dirk+"kerr.dat."+str(it), np.transpose([kerr.real, fara, kerrR]))
  
  # update time
  it = it + 1
  fim = "../probe_results/imEps." + str(it) + ".dat"
