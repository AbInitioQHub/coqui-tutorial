#!/bin/bash
#SBATCH -p ccq
#SBATCH -C genoa
#SBATCH -t 8:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -c 1
#SBATCH -o svo_ase.o%j
#SBATCH -J svo_ase

module load llvm19-mkl-mod2.4 quantum-espresso triqs-collections
source ~/py_env/py311_mod2.4/bin/activate

#OpenMP settings:
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

date
python svo_ase.py 
date

