import csv
import cv2
import numpy as np
from pathlib import Path
from skimage.feature import hog

# Import your existing color feature module
from color_features import (
    DEFAULT_HIST_BINS,
    extract_color_features,
    feature_columns as color_feature_columns
)
from hog_feature import extract_hog_features, hog_feature_columns
# --- Configuration ---
DATA_DIR = Path(__file__).parent / "Data"
OUTPUT_CSV = Path(__file__).parent / "dataset.csv"

LABELS = {
    "ai_images": "ai",
    "real_art": "real",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}




ACTIVE_MODULES = {
    "color": {
        "get_columns": lambda: color_feature_columns(DEFAULT_HIST_BINS),
        "extract": lambda img_path: extract_color_features(img_path, hist_bins=DEFAULT_HIST_BINS),
    },
    "hog": {
        "get_columns": lambda: hog_feature_columns(),
        "extract": lambda img_path: extract_hog_features(img_path),
    }
}


def get_all_expected_columns():
    """Dynamically builds the CSV header based on active modules."""
    cols = ["filepath", "label"]
    for mod_name, mod_data in ACTIVE_MODULES.items():
        cols.extend(mod_data["get_columns"]())
    return cols


def collect_image_rows(data_dir=DATA_DIR, labels=LABELS, image_exts=IMAGE_EXTS):
    data_dir = Path(data_dir)
    rows = []
    for folder_name, label in labels.items():
        folder = data_dir / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in image_exts:
                rel = path.relative_to(data_dir).as_posix()
                rows.append((rel, label))
    return rows


def load_processed(csv_path, expected_columns):
    """
    STRICT CHECK: Only skips a file if its row contains valid data
    for ALL currently expected columns.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return set()

    done = set()
    expected_set = set(expected_columns)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        csv_headers = set(reader.fieldnames or [])

        # Warn if the CSV is completely missing columns we currently expect
        if not expected_set.issubset(csv_headers):
            print("Notice: CSV is missing some columns required by current active modules.")
            print("Files missing these new columns will be re-processed.\n")

        for row in reader:
            # Check if every expected column has a non-empty value in this row
            has_all_features = all(row.get(col) not in [None, ""] for col in expected_columns)
            if has_all_features:
                done.add(row["filepath"])

    return done


def build_dataset(
        data_dir=DATA_DIR,
        csv_path=OUTPUT_CSV,
        labels=LABELS,
        image_exts=IMAGE_EXTS,
        progress_every=100,
):
    data_dir = Path(data_dir)
    csv_path = Path(csv_path)

    # 1. Get dynamic columns based on active modules
    columns = get_all_expected_columns()

    # 2. Find which files actually need processing (strict check)
    done = load_processed(csv_path, columns)
    all_rows = collect_image_rows(data_dir, labels, image_exts)
    pending = [r for r in all_rows if r[0] not in done]

    file_exists = csv_path.exists()
    added = 0
    errors = []

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()

        for idx, (rel, label) in enumerate(pending, 1):
            img_path = data_dir / rel
            row = {"filepath": rel, "label": label}

            try:
                # 3. Dynamically run all active modules
                for mod_name, mod_data in ACTIVE_MODULES.items():
                    feats = mod_data["extract"](img_path)
                    row.update(feats)

            except Exception as e:
                errors.append((rel, f"[{mod_name} module error] {type(e).__name__}: {e}"))
                continue

            # 4. Write the fully constructed row
            writer.writerow(row)
            added += 1

            if progress_every and idx % progress_every == 0:
                print(f"  {idx}/{len(pending)} processed (errors so far: {len(errors)})")
                f.flush()

    return {
        "added": added,
        "skipped": len(all_rows) - len(pending),
        "errors": errors,
        "csv_path": csv_path,
    }


def main():
    print(f"Starting Extraction Pipeline...")
    print(f"Active Modules: {list(ACTIVE_MODULES.keys())}")

    result = build_dataset()

    print(f"\nAdded/Updated {result['added']} rows to {result['csv_path']}")
    print(f"Skipped {result['skipped']} already fully processed files")

    if result["errors"]:
        print(f"\nErrors: {len(result['errors'])} (showing first 5)")
        for path, err in result["errors"][:5]:
            print(f"  {path}: {err}")


if __name__ == "__main__":
    main()