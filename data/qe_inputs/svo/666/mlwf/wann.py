from mpi4py import MPI
import coqui

# mpi handler and verbosity
mpi = coqui.MpiHandler()
coqui.set_verbosity(mpi, output_level=2)

# construct MF from a dictionary 
mf_params = {
    "prefix": "svo", 
    "outdir": "../out",
    "nbnd": 40
}
mf = coqui.make_mf(mpi, mf_params, "qe")

# wannier90
w90_params = {
  "prefix": "svo",
  "shells": {
    "atoms": [0],
    "sort": [0],
    "l": [2],
    "dim": [3],
    "SO": [0],
    "irep": [0]
  }
}
coqui.wannier90(mf, w90_params)
