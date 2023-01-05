#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import scipy.optimize

fin="sz_elec_tot.out"
tstart = 0 # ps
tend = np.inf # ps. if < 0, means +inf
Bfield = 0 # in-plane (for Sz) magnetic field in Tesla

t_data = np.loadtxt(fin,usecols=(0,))
nt = t_data.shape[0]
t_data = t_data * 2.4188843265857e-5 # / 40000. # a.u. to ps
mask = (t_data >= tstart) & (t_data <= tend)
t_data = t_data[mask]
print("Fitting range: [",t_data[0],", ",t_data[-1],"] ps")
t_data = t_data - t_data[0]
s_data = np.loadtxt(fin,usecols=(1,))[0:nt][mask]

# ==== theoretical parameter values ====
beta = 1 / 100. # initial guess of rate in 1/ps
Bfield = Bfield / 2.3505175675871e5 # convert to a.u.
omega = Bfield / 2.4188843265857e-5
phi = 0 # phase
params = beta, omega, phi

# ==== model ====
def decay(t, beta):
  s = s_data[0] * np.exp(-beta*t_data)
  return s
def decay_cosine(t, beta, omega, phi):
  s = s_data[0] * np.exp(-beta*t_data) * np.cos(omega*t_data + phi)
  return s
def residuals1(args, t, s):
  return s - decay(t, *args)
def residuals2(args, t, s):
  return s - decay_cosine(t, *args)

# ==== fitting using curve_fit ====
if omega == 0:
  params_cf, _ = scipy.optimize.curve_fit(decay, t_data, s_data)
  params_lsq, _ = scipy.optimize.leastsq(residuals1, beta, args=(t_data, s_data))
else:
  params_cf, _ = scipy.optimize.curve_fit(decay_cosine, t_data, s_data)
  params_lsq, _ = scipy.optimize.leastsq(residuals2, params, args=(t_data, s_data))

print("Global exponential fit by two ways:")
if omega == 0:
  print("cf: tau = ",1/params_cf[0]," ps")
  print("lsq: tau = ",1/params_lsq[0]," ps")
else:
  print("cf: tau = ",1/params_cf[0]," ps"," period = ",2*np.pi/params_cf[1],"ps"," phi = ",params_cf[2])
  print("lsq: tau = ",1/params_lsq[0]," ps"," period = ",2*np.pi/params_lsq[1],"ps"," phi = ",params_lsq[2])

if omega == 0:
  s_fit = decay(t_data, params_lsq[0])
else:
  s_fit = decay_cosine(t_data, params_lsq[0], params_lsq[1], params_lsq[2])
np.savetxt("sfit.out", np.transpose([t_data, s_data, s_fit]))

#log fit
s_data = np.log(np.abs(s_data))
nt = t_data.shape[0]
t1 = np.zeros(nt-1)
for it in range(2,nt+1):
  fit = np.polyfit(t_data[it-2:it],s_data[it-2:it],1)
  t1[it-2] = -1. / fit[0] # ps

print("Log fit of time-resolved spin lifetime:")
if nt-1 > 20:
  print(t1[0:10])
  print(t1[nt-11:nt-1])
else:
  print(t1)
np.savetxt("tau_t.dat",np.transpose([t_data[0:nt-1],t1]))
