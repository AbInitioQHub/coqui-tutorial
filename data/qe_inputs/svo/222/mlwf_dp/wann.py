from mpi4py import MPI
import coqui

import matplotlib.pyplot as plt

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
  "h5_filename": "svo_t2g.mlwf.h5",
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

# Interpolate bands with the disentangled Wannier file
winter_params_dp = {
    "outdir": "./",
    "prefix": "svo",
    "wannier_file": "svo_dp.mlwf.h5",
    "kpath": """
      G 0.00 0.00 0.00
      X 0.00 0.50 0.00
      M 0.50 0.50 0.00
      G 0.00 0.00 0.00
    """,
}
coqui.post_proc.band_interpolation(mf, winter_params_dp)

if mpi.root():
    fig, ax = plt.subplots(1, figsize=(8, 5), dpi=80)

    coqui.post_proc.band_plot(
        ax,
        "svo.mbpt.h5",
        color="tab:red",
        linestyle="-",
        linewidth=2.0,
        label="PBE",
        fontsize=14,
    )

    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5, alpha=0.5)
    ax.legend(loc=1, fontsize=12)
    plt.tight_layout()

    plt.savefig("svo_dft.png", format="png")
