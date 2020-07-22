#!/usr/bin/env python
import numpy as np

e = np.fromfile("bandstruct.eigenvals", dtype=np.float64).reshape(260,44)
de = e[:,29] - e[:,28]
np.savetxt("de.dat",de)

sfull = np.fromfile("bandstruct.S", dtype=np.complex128).reshape((260,3,44,44))
s = sfull[:,0:3,28:30,28:30]
sz = s[:,2,:,:]

np.savetxt("szc_dft_c1.dat",np.real(sz[0:259,0,0]))
np.savetxt("szc_dft_c2.dat",np.real(sz[0:259,1,1]))
