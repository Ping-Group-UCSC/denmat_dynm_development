#!/usr/bin/env python
import numpy as np

t = np.loadtxt("Sz_elec_tot.out",usecols=(0,))
t = t * 2.4188843265857e-5 # / 40000. # a.u. to ps
sz = np.loadtxt("Sz_elec_tot.out",usecols=(1,))
sz = np.log(np.abs(sz))

nt = t.shape[0]
t1 = np.zeros(nt-1)

for it in range(2,nt+1):
  fit = np.polyfit(t[it-2:it],sz[it-2:it],1)
  t1[it-2] = -1. / fit[0] # ps

print(t1)
np.savetxt("tau(t).dat",np.transpose([t[0:nt-1],t1]))

fit = np.polyfit(t[0:nt],sz[0:nt],1)
t1_tot = -1. / fit[0] # ps

print(t1_tot)
