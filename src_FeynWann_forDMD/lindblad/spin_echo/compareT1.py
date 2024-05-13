#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os

ps = 1000. #in fs
EhInv = 0.024188  #Eh^-1 in fs

for iDir, dirName in enumerate("xyz"):
    fname = f'T1{dirName}.out'
    if not os.path.isfile(fname):
        continue
    t = []
    Sdir = []
    for line in open(fname):
        if line.startswith('tauSpinEY [fs]'):
            tauExpected = float(line.split()[4 + iDir])
        if line.startswith('Integrate: Step:'):
            tokens = line.split()
            t.append(float(tokens[4]))
            Sdir.append(float(tokens[11 + iDir]))
    t = np.array(t[1:])
    Sdir = np.array(Sdir[1:])
    plt.plot(t/ps, Sdir/Sdir[0], label=f"$S_{dirName}$")
    plt.plot(t/ps, np.exp(-t/tauExpected), 'k:', lw=1)
plt.ylabel(fr'$\langle S_{"xyz"[iDir]}(t) \rangle$')
plt.xlabel(r'$t$ [ps]')
plt.xlim(t[0]/ps, t[-1]/ps)
plt.ylim(None, 1.0)
plt.yscale("log")
plt.savefig(f'T1.pdf', bbox_inches='tight')
plt.legend()
plt.show()
