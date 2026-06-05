import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "qe_inputs" / "svo" / "222"

COPY_TARGETS = [
    (
        DATA_DIR / "out",
        NOTEBOOK_DIR / "out",
    ),
    (
        DATA_DIR / "svo.pw2coqui.in",
        NOTEBOOK_DIR / "svo.pw2coqui.in",
    ),
    (
        DATA_DIR / "mlwf" / "svo.win",
        NOTEBOOK_DIR / "mlwf" / "svo.win",
    ),
    (
        DATA_DIR / "mlwf_dp" / "svo.win",
        NOTEBOOK_DIR / "mlwf_dp" / "svo.win",
    ),
]

EXCLUDED_FILENAMES = {
#    NOTEBOOK_DIR / "out": {"svo.coqui.h5"},
}

LEGACY_TARGETS = [
    NOTEBOOK_DIR / "out",
    NOTEBOOK_DIR / "svo.pw2coqui.in",
    NOTEBOOK_DIR / "mlwf",
    NOTEBOOK_DIR / "mlwf_dp",
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
            shutil.copytree(
                source,
                destination,
                ignore=shutil.ignore_patterns(*EXCLUDED_FILENAMES.get(destination, set())),
            )
        else:
            shutil.copy2(source, destination)
        print(f"Copied: {source.relative_to(ROOT)} -> {destination.relative_to(NOTEBOOK_DIR)}")


def main() -> None:
    remove_legacy_targets()
    copy_inputs()
    print("Notebook inputs are ready.")


if __name__ == "__main__":
    main()
