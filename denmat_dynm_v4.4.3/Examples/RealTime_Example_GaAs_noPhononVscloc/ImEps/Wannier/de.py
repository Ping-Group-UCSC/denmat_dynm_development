#!/usr/bin/env python
import numpy as np

e1 = np.loadtxt("wannier.eigenvals", dtype=np.float64, usecols=(6))
e2 = np.loadtxt("wannier.eigenvals", dtype=np.float64, usecols=(7))
de = e2 - e1
np.savetxt("de.dat",de)
print("from L to G\n",de[115:185])
print("from G to K\n",de[185:259])
