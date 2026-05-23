import triqs.utility.mpi as mpi
import triqs_modest as modest

import coqui

coqui_mpi = coqui.MpiHandler()
coqui.set_verbosity(coqui_mpi, output_level=1)

# Define correlated subspace via ModEST
wan_h5 = "nio_555/mlwf_dp/nio_eg.mlwf.h5"
obe = modest.make_one_body_elements_gw(wan_h5)
E1 = modest.make_embedding(obe.C_space)

mf_params = {"prefix": "nio", "outdir": "nio_555/out", "nbnd": 40}
mf = coqui.make_mf(coqui_mpi, params=mf_params, mf_type='qe')

thc_params = {"thresh": 1e-5}
thc = coqui.make_thc_coulomb(mf=mf, params=thc_params)

gw_edmft_params = {
  "outdir": "./",
  "prefix": "nio",
  "niter": 1,
  "wannier_file": wan_h5,
  "screen_type": "rpa",
  "edmft": {
    "chkpt_h5": "nio.dmft.h5",
    "impurity": [
        {
         "degenerate_blk_thresh": 0.001,
         "length_cycle": 200,
         "n_warmup_cycles": 5e4,
         "n_cycles": 1e6,
         "perform_tail_fit": True,
         "fit_max_moment": 9,
         "fit_min_w": 2.0,
         "fit_max_w": 10.0,
         "chemical_potential": {"tolerance": 0.05}
       }
    ]
  }
}
coqui.dmft.run_gw_edmft(thc, E1, params=gw_edmft_params)
