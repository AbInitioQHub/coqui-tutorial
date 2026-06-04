import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = Path(__file__).resolve().parent

SI_555_DIR = ROOT / "data" / "qe_inputs" / "si" / "555"
#SI_222_DIR = ROOT / "data" / "qe_inputs" / "si" / "222"
#SI_777_DIR = ROOT / "data" / "qe_inputs" / "si" / "777"
#COQUI_777_DIR = ROOT / "data" / "coqui" / "si" / "777"

COPY_TARGETS = [
    (SI_555_DIR / "out", NOTEBOOK_DIR / "si_555" / "out"),
    (SI_555_DIR / "mlwf" / "si.win", NOTEBOOK_DIR / "si_555" / "mlwf" / "si.win"),
    (SI_555_DIR / "mlwf" / "si.mlwf.h5", NOTEBOOK_DIR / "si_555" / "mlwf" / "si.mlwf.h5"),
    #(SI_222_DIR / "out", NOTEBOOK_DIR / "si_222" / "out"),
    #(SI_222_DIR / "mlwf" / "si.win", NOTEBOOK_DIR / "si_222" / "mlwf" / "si.win"),
    #(SI_222_DIR / "mlwf" / "si.mlwf.h5", NOTEBOOK_DIR / "si_222" / "mlwf" / "si.mlwf.h5"),
    #(SI_777_DIR / "out", NOTEBOOK_DIR / "si_777" / "out"),
    #(SI_777_DIR / "mlwf" / "si.mlwf.h5", NOTEBOOK_DIR / "si_777" / "mlwf" / "si.mlwf.h5"),
    #(SI_777_DIR / "mlwf" / "si.win", NOTEBOOK_DIR / "si_777" / "mlwf" / "si.win"),
    #(COQUI_777_DIR / "si.mbpt.h5", NOTEBOOK_DIR / "si_777" / "coqui" / "si.mbpt.h5"),
    #(COQUI_777_DIR / "si_qpg0w0.mbpt.h5", NOTEBOOK_DIR / "si_777" / "coqui" / "si_qpg0w0.mbpt.h5"),
]

LEGACY_TARGETS = [
    #NOTEBOOK_DIR / "si_222",
    #NOTEBOOK_DIR / "si_777",
    NOTEBOOK_DIR / "si_555",
]


def remove_legacy_targets() -> None:
    for target in LEGACY_TARGETS:
        if target.is_dir():
            shutil.rmtree(target)
            print(f"Removed directory: {target.relative_to(NOTEBOOK_DIR)}")
        elif target.is_file() or target.is_symlink():
            target.unlink()
            print(f"Removed file: {target.relative_to(NOTEBOOK_DIR)}")


def copy_inputs() -> None:
    for source, destination in COPY_TARGETS:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        print(f"Copied: {source.relative_to(ROOT)} -> {destination.relative_to(NOTEBOOK_DIR)}")


def main() -> None:
    remove_legacy_targets()
    copy_inputs()
    print("Notebook inputs are ready.")


if __name__ == "__main__":
    main()
