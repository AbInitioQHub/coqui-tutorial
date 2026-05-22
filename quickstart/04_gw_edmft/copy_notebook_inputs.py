import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "qe_inputs" / "svo"

COPY_TARGETS = [
    (DATA_DIR / "666", NOTEBOOK_DIR / "svo_666"),
    (ROOT / "data" / "coqui" / "svo" / "666" / "svo.mbpt.h5", NOTEBOOK_DIR / "svo.mbpt.h5"),
]

LEGACY_TARGETS = [
    NOTEBOOK_DIR / "svo_666",
    NOTEBOOK_DIR / "svo.mbpt.h5",
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
