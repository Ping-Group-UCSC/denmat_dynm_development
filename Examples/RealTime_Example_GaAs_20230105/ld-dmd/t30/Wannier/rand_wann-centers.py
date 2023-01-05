#!/usr/bin/env python
import os
import random

random.seed()

f = open("rand_wann-centers.dat","w")

#for i in range(7):
#    for spin in ("sUp", "sDn"):
#        f.write("wannier-center Gaussian %10.6f %10.6f %10.6f 2.0 %s \n" %
#                (random.random()-0.5, random.random()-0.5, random.random()-0.5, spin) )
for i in range(8):
  x=random.random()-0.5
  y=random.random()-0.5
  z=random.random()-0.5
  f.write("wannier-center Gaussian %10.6f %10.6f %10.6f 1.7 %s \n" % (x,y,z,"sUp"))
  f.write("wannier-center Gaussian %10.6f %10.6f %10.6f 1.7 %s \n" % (x,y,z,"sDn"))
