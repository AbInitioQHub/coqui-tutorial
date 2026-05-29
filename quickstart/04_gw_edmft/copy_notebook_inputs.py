import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "qe_inputs" / "nio"

COPY_TARGETS = [
    (DATA_DIR / "555", NOTEBOOK_DIR / "nio_555"),
    (ROOT / "data" / "coqui" / "nio" / "555", NOTEBOOK_DIR / "coqui_chkpts"),
]

LEGACY_TARGETS = [
    NOTEBOOK_DIR / "nio_555",
    NOTEBOOK_DIR / "coqui_chkpts",
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
