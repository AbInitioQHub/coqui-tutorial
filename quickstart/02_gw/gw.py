from mpi4py import MPI
import coqui

# Create CoQui MPI handler and set logging verbosity in the beginning
coqui_mpi = coqui.MpiHandler()
coqui.set_verbosity(coqui_mpi, output_level=1)

# Step 1: Build mean-field object
mf_params = {
    "prefix": "si",
    "outdir": "si_555/out",
    "nbnd": 20,
}
mf = coqui.make_mf(coqui_mpi, params=mf_params, mf_type="qe")

# Step 2: Build THC Coulomb Hamiltonian
thc_params = {
    "ecut": 1.2 * mf.ecutwfc(),
    "thresh": 1e-3,
    "save": "thc.coulomb.h5"
}
thc = coqui.make_thc_coulomb(mf=mf, params=thc_params)

# Step 3: Run GW calculation
gw_params = {
    "outdir": "./",
    "prefix": "si.gw",
    "niter": 6,
    "beta": 700,
    "iaft": {"eps": 1e-8}
}
coqui.run_gw(params=gw_params, h_int=thc)
