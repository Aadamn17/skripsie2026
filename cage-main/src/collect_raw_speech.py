"""Copy raw speech audio for patients with at least one speech recording."""
import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "cage" / "counting"
DEFAULT_DESTINATION_DIR = PROJECT_ROOT / "data" / "cage" / "raw_speech"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy speech WAV files for patients whose counting folder "
            "contains at least one WAV file."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Speech source directory (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--destination-dir",
        type=Path,
        default=DEFAULT_DESTINATION_DIR,
        help=f"Output directory (default: {DEFAULT_DESTINATION_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report matching patients and files without copying anything.",
    )
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        parser.error(f"Source directory does not exist: {args.source_dir}")

    patient_files = {
        patient_dir: sorted(
            path for path in patient_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".wav"
        )
        for patient_dir in sorted(args.source_dir.iterdir())
        if patient_dir.is_dir()
    }
    patient_files = {
        patient_dir: files
        for patient_dir, files in patient_files.items()
        if files
    }
    file_count = sum(len(files) for files in patient_files.values())

    print(f"Patients with speech WAV files: {len(patient_files)}")
    print(f"Speech WAV files to copy: {file_count}")
    if args.dry_run:
        print("Dry run; no files copied.")
        return

    for patient_dir, files in patient_files.items():
        destination_patient_dir = args.destination_dir / patient_dir.name
        destination_patient_dir.mkdir(parents=True, exist_ok=True)
        for source_file in files:
            shutil.copy2(source_file, destination_patient_dir / source_file.name)

    print(f"Copied speech files to: {args.destination_dir}")


if __name__ == "__main__":
    main()