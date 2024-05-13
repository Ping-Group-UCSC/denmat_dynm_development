#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import glob

ps = 1000. #in fs
EhInv = 0.024188  #Eh^-1 in fs

styles=['solid','solid', 'solid', 'solid','solid', 'solid', 'solid']
for iFile, fname in enumerate(glob.glob('lindblad*.out')):
	fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
	plt.subplots_adjust(hspace=0.1)
	title = fname.replace('.out','')
	print(f'Plotting {title}')
	t = []
	S = []
	Sprime = []
	for line in open(fname):
		if line.startswith('spinEchoDelay ='):
			tDelay = float(line.split()[-1]) * EhInv
		if line.startswith('spinEchoFlipTime ='):
			tFlip = float(line.split()[2]) * EhInv
		if line.startswith('Integrate: Step:'):
			tokens = line.split()
			t.append(float(tokens[4]))
			S.append([float(tok) for tok in tokens[11:14]])
		if line.startswith('SpinEcho: Srot:'):
			Sprime.append([float(tok) for tok in line.split()[3:6]])
	t = np.array(t[1:])
	S = np.array(S[1:]).T
	Sprime = np.array(Sprime[1:]).T
	
	#Plot each direction in separate panel:
	for iDir, ax in enumerate(axes):
		plt.sca(ax)
		plt.plot(t/ps, S[iDir], label='Lab frame')
		plt.plot(t/ps, Sprime[iDir], label='Rotating frame')
		plt.axvline((tDelay + 0.5*tFlip)/ps, linestyle='--', color='black')
		plt.axvline((2*tDelay + 0.75*tFlip)/ps, linestyle='--', color='black')
		plt.ylabel(fr'$\langle S_{"xyz"[iDir]}(t) \rangle$')
	plt.xlabel(r'$t$ [ps]')
	plt.xlim(t[0]/ps, t[-1]/ps)
	plt.savefig(f'{title}.png', bbox_inches='tight')
	plt.legend()
plt.show()
