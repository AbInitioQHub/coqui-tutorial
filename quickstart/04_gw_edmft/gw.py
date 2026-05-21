import triqs.utility.mpi as mpi
import coqui

coqui_mpi = coqui.MpiHandler()
coqui.set_verbosity(coqui_mpi, output_level=2)

mf_params = {"prefix": "svo", "outdir": "svo_666/out", "nbnd": 40}
mf = coqui.make_mf(
    coqui_mpi,
    params=mf_params,
    mf_type='qe'
)

thc_params = {"thresh": 1e-3, "save": "thc.coulomb.h5"}
thc = coqui.make_thc_coulomb(mf=mf, params=thc_params)

# self-consistent GW as starting point
gw_params = {
    "outdir": "./",
    "prefix": "svo",
    "beta": 100,
    "niter": 8,
    "div_treatment": "gygi_metal",
    "iter_alg": {
        "alg": "diis",
        "mixing": 0.4,
        "max_subsp_size": 5,
        "diis_warmup": 3
    }
}
coqui.run_gw(params=gw_params, h_int=thc)
