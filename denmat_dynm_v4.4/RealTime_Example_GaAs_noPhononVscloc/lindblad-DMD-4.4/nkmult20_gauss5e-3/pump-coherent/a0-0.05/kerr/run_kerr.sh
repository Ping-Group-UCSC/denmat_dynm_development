#!/bin/bash
#SBATCH -p debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=dm
#SBATCH --output=job%j.out

python kerr.py

for e in 1.41 1.42 1.43 1.45 1.47 1.49 1.51
do
  echo $e
  echo $e > ftmp
  python kerrt_atE.py < ftmp
done
rm ftmp
