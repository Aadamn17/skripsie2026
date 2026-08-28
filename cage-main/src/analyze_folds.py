"""Report patient, label, and cough counts for each data fold."""

import argparse
from pathlib import Path

import pandas as pd


def summarize_fold(fold_path: Path) -> dict[str, int]:
    """Return patient and cough counts for one fold CSV."""
    fold = pd.read_csv(fold_path)
    required_columns = {"Cough_ID", "Status"}
    missing_columns = required_columns - set(fold.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{fold_path} is missing required column(s): {missing}")

    fold["Patient_ID"] = fold["Cough_ID"].astype(str).str.split("/").str[0]
    patient_statuses = fold.groupby("Patient_ID")["Status"].nunique()
    inconsistent_patients = patient_statuses[patient_statuses > 1]
    if not inconsistent_patients.empty:
        patients = ", ".join(inconsistent_patients.index)
        raise ValueError(f"{fold_path} has mixed statuses for patient(s): {patients}")

    patient_labels = fold.groupby("Patient_ID")["Status"].first()
    return {
        "patients": int(patient_labels.size),
        "positive": int((patient_labels == 1).sum()),
        "negative": int((patient_labels == 0).sum()),
        "coughs": int(len(fold)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate patient, positive/negative, and cough counts per fold."
    )
    parser.add_argument(
        "--fold-dir",
        type=Path,
        default=Path("data/cage/data_folds_filtered"),
        help="Directory containing fold_*.csv files.",
    )
    args = parser.parse_args()

    fold_paths = sorted(args.fold_dir.glob("fold_*.csv"))
    if not fold_paths:
        parser.error(f"No fold_*.csv files found in {args.fold_dir}")

    print(f"Fold directory: {args.fold_dir}")
    print("fold,patients,positive,negative,coughs")
    totals = {"patients": 0, "positive": 0, "negative": 0, "coughs": 0}
    for fold_path in fold_paths:
        counts = summarize_fold(fold_path)
        print(
            f"{fold_path.stem},{counts['patients']},{counts['positive']},"
            f"{counts['negative']},{counts['coughs']}"
        )
        for key in totals:
            totals[key] += counts[key]

    print(
        f"total,{totals['patients']},{totals['positive']},"
        f"{totals['negative']},{totals['coughs']}"
    )


if __name__ == "__main__":
    main()