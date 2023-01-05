#!/usr/bin/env python
import numpy as np

e1 = np.loadtxt("wannier.eigenvals", dtype=np.float64, usecols=(6))
e2 = np.loadtxt("wannier.eigenvals", dtype=np.float64, usecols=(7))
de = e2 - e1
np.savetxt("de.dat",de)
print("from G to X\n",de[0:11])
print("from L to G\n",de[448:459])
print("from G to K\n",de[458:470])
