#!/usr/bin/env python
import numpy as np

e = np.fromfile("bandstruct.eigenvals", dtype=np.float64).reshape(643,34)
np.savetxt("ecbm.dat", e[:,8] - e[458,8])
de = e[448:470,9] - e[448:470,8]
np.savetxt("de.dat",de)
de = e[0:11,9] - e[0:11,8]
np.savetxt("deGX.dat",de)

sfull = np.fromfile("bandstruct.S", dtype=np.complex128).reshape((643,3,34,34))
s = sfull[:,0:3,8:10,8:10]
sz = s[:,2,:,:]

np.savetxt("szc_dft_c1.dat",np.real(sz[448:470,0,0]))
np.savetxt("szc_dft_c2.dat",np.real(sz[448:470,1,1]))
