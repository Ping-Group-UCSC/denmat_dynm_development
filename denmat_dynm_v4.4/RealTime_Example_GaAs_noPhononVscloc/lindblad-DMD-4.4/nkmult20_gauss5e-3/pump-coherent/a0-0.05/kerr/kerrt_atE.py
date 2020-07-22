#!/usr/bin/env python
import numpy as np
import os.path
import subprocess

esel = input("Enter probe energy: ")
dirk = "kerr_results/"
#dirk = "kerr2D_results/"
#dirk = "kerr2DonSiO2_results/"
time = np.loadtxt("../Sz_elec_tot.out", usecols=(0,))
time = np.delete(time, 1)
nt = time.shape[0]

e=np.loadtxt(dirk+"omega.dat")
ie = np.argwhere(np.abs(e - esel) < 1e-6)[0,0]

kt = np.zeros(nt)
ft = np.zeros(nt)
kRt = np.zeros(nt)

for it in range(nt):
  fim = dirk + "kerr.dat." + str(it)
  ke = np.loadtxt(fim, usecols=(0,))
  fe = np.loadtxt(fim, usecols=(1,))
  kRe = np.loadtxt(fim, usecols=(2,))
  kt[it] = ke[ie]
  ft[it] = fe[ie]
  kRt[it] = kRe[ie]

np.savetxt(dirk+"kerrt_at"+str(esel)+".dat", np.transpose([time,kt,ft,kRt]))
