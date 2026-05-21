import triqs.utility.mpi as mpi
import triqs_modest as modest

import coqui

coqui_mpi = coqui.MpiHandler()
coqui.set_verbosity(coqui_mpi, output_level=1)

# Define correlated subspace via ModEST
wan_h5 = "svo_666/mlwf/svo.mlwf.h5"
obe = modest.make_one_body_elements_gw(wan_h5)
E1 = modest.make_embedding(obe.C_space)

mf_params = {"prefix": "svo", "outdir": "svo_666/out", "nbnd": 40}
mf = coqui.make_mf(coqui_mpi, params=mf_params, mf_type='qe')

thc_params = {"thresh": 1e-3, "save": "thc.coulomb.h5"}
thc = coqui.make_thc_coulomb(mf=mf, params=thc_params)

gw_edmft_params = {
  "outdir": "./",
  "prefix": "svo",
  "niter": 10,
  "wannier_file": wan_h5,
  "screen_type": "gw_edmft",
  "div_treatment": "gygi_metal",
  "edmft": {
    "chkpt_h5": "svo.dmft.h5",
    "impurity": [
        {
         "degenerate_blk_thresh": 0.001,
         "length_cycle": 500,
         "n_warmup_cycles": 1e4,
         "n_cycles": 4e6,
         "perform_tail_fit": True,
         "fit_max_moment": 20,
         "fit_min_w": 2.0,
         "fit_max_w": 10.0,
         "chemical_potential": {"tolerance": 0.001}
       }
    ]
  }
}
coqui.dmft.run_gw_edmft(thc, E1, params=gw_edmft_params)

