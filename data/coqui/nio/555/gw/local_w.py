from mpi4py import MPI
import coqui

coqui_mpi = coqui.MpiHandler()
coqui.set_verbosity(coqui_mpi, output_level=2)

qe_dir = "../../../../qe_inputs/nio/555/"

mf_params = {"prefix": "nio", "outdir": qe_dir+"out", "nbnd": 40}
mf = coqui.make_mf(
    coqui_mpi,
    params=mf_params,
    mf_type='qe'
)

thc_params = {"thresh": 1e-3}
thc = coqui.make_thc_coulomb(mf=mf, params=thc_params)

rpa_params = {
    "screen_type": "rpa",
    "greens_func_source": "scf",
    "greens_func_iteration": 1,
    "prefix": "nio",
    "wannier_file": qe_dir+"/mlwf_dp/nio_eg.mlwf.h5",
}
Vloc, Uloc_iw = coqui.downfold_coulomb(h_int=thc, params=rpa_params)

rpa_params["greens_func_iteration"] = -1
Vloc, Uloc_iw = coqui.downfold_coulomb(h_int=thc, params=rpa_params)


