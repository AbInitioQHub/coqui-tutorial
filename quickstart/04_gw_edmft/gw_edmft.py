import triqs.utility.mpi as mpi
import triqs_modest as modest

import coqui
import coqui.dmft

import time

coqui_mpi = coqui.MpiHandler()
coqui.set_verbosity(coqui_mpi, output_level=1)

coqui.app_log(1, f"{time.ctime()}")

# Define correlated subspace via ModEST
wan_h5 = "nio_555/mlwf_dp/nio_eg.mlwf.h5"
obe = modest.make_one_body_elements_gw(wan_h5)
embedding = modest.make_embedding(obe.C_space)

# Construct Mf 
mf_params = {"prefix": "nio", "outdir": "nio_555/out", "nbnd": 40}
mf = coqui.make_mf(coqui_mpi, params=mf_params, mf_type='qe')

# Construct THC Coulomb Hamiltonian
thc_params = {
  "ecut": 1.2 * mf.ecutwfc(),
  "thresh": 1e-3, 
  "save": "thc.coulomb.h5"
}
thc = coqui.make_thc_coulomb(mf=mf, params=thc_params)

# GW+EDFMT 
gw_edmft_params = {
  "outdir": "./coqui_chkpts/gw/",
  "prefix": "nio",
  "niter": 4,
  "wannier_file": wan_h5,
  "screen_type": "rpa",
  "iter_alg": {"alg": "damping", "mixing": 0.7},
  "edmft": {
    "chkpt_h5": "nio.dmft.h5",
    "impurity": {
      "degenerate_blk_thresh": 0.001,
      "length_cycle": 50,
      "n_warmup_cycles": 5e4,
      "n_cycles": 5e5,
      "perform_tail_fit": True,
      "fit_max_moment": 9,
      "fit_min_w": 2.0,
      "fit_max_w": 10.0,
      "chemical_potential": {
         "tolerance": 0.2, 
         "n_cycles": 1e3
      }
    }
  }
}
coqui.dmft.run_gw_edmft(thc, embedding, params=gw_edmft_params)
coqui.app_log(1, f"{time.ctime()}")