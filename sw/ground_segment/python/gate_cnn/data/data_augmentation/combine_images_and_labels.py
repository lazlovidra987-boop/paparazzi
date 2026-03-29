import os
import sys
import json
import shutil
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# =========================================
# OUTPUT DATASET (staat boven dit script)
# =========================================
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DATASET_DIR = SCRIPT_DIR / "combined_dataset"
OUTPUT_IMAGES_DIR  = OUTPUT_DATASET_DIR / "images"
OUTPUT_LABELS_PATH = OUTPUT_DATASET_DIR / "labels.json"


# =========================================
# HELPERS
# =========================================

def choose_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Selecteer dataset folder om toe te voegen")
    return Path(folder) if folder else None


def find_labels_file(folder: Path) -> Path | None:
    candidates = [
        folder / "labels.json",
        folder / "labels",
    ]
    for c in candidates:
        if c.is_file():
            return c
    for f in folder.glob("*.json"):
        if "label" in f.name.lower():
            return f
    return None


def find_images_dir(folder: Path) -> Path | None:
    for name in ("images", "image", "imgs"):
        candidate = folder / name
        if candidate.is_dir():
            return candidate
    return None


def load_labels(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Labels file is geen lijst: {path}")
    return data


def save_labels(data: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def make_unique_name(filename: str, used_names: set, prefix: str = "") -> str:
    name, ext = os.path.splitext(filename)
    candidate = f"{prefix}{filename}"
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate
    counter = 1
    while True:
        candidate = f"{prefix}{name}_{counter}{ext}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        counter += 1


# =========================================
# MAIN
# =========================================

def main():
    # --- Bepaal input folder ---
    if len(sys.argv) > 1:
        input_dir = Path(sys.argv[1])
        if not input_dir.exists():
            print(f"Fout: folder bestaat niet: {input_dir}")
            return
    else:
        print("Selecteer de dataset folder die je wil toevoegen ...")
        input_dir = choose_folder()
        if input_dir is None:
            print("Geen folder geselecteerd. Stop.")
            return

    # --- Zoek images en labels in input folder ---
    images_dir  = find_images_dir(input_dir)
    labels_file = find_labels_file(input_dir)

    if images_dir is None:
        print(f"Fout: geen images-map gevonden in {input_dir}")
        return
    if labels_file is None:
        print(f"Fout: geen labels file gevonden in {input_dir}")
        return

    print(f"Input folder  : {input_dir}")
    print(f"Images folder : {images_dir}")
    print(f"Labels file   : {labels_file}")

    # --- Maak output mappen aan als die nog niet bestaan ---
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Laad bestaande labels (of begin leeg) ---
    if OUTPUT_LABELS_PATH.exists():
        existing_labels = load_labels(OUTPUT_LABELS_PATH)
        print(f"\nBestaande entries in output: {len(existing_labels)}")
    else:
        existing_labels = []
        print("\nNog geen output labels gevonden, begin nieuw.")

    # Bouw set van al gebruikte bestandsnamen zodat we geen dubbelen maken
    used_names = {item["img_name"] for item in existing_labels if "img_name" in item}

    # --- Laad nieuwe labels ---
    new_labels = load_labels(labels_file)

    # Gebruik mapnaam als prefix om botsingen met andere datasets te vermijden
    prefix = f"{input_dir.name}_"

    added   = 0
    skipped = 0

    for item in new_labels:
        if "img_name" not in item:
            print(f"Overgeslagen (geen img_name): {item}")
            skipped += 1
            continue

        old_name = item["img_name"]
        src_path = images_dir / old_name

        if not src_path.is_file():
            print(f"Afbeelding niet gevonden, skip: {src_path}")
            skipped += 1
            continue

        new_name = make_unique_name(old_name, used_names, prefix=prefix)
        dst_path = OUTPUT_IMAGES_DIR / new_name

        shutil.copy2(src_path, dst_path)

        new_item = {
            "img_name": new_name,
            "heading_angle": item.get("heading_angle", 0.0),
            "confidence": item.get("confidence", 0.0),
        }
        existing_labels.append(new_item)
        added += 1

    # --- Sla samengevoegde labels op ---
    save_labels(existing_labels, OUTPUT_LABELS_PATH)

    print(f"\nKlaar.")
    print(f"Toegevoegd    : {added}")
    print(f"Overgeslagen  : {skipped}")
    print(f"Totaal entries: {len(existing_labels)}")
    print(f"Output images : {OUTPUT_IMAGES_DIR}")
    print(f"Output labels : {OUTPUT_LABELS_PATH}")


if __name__ == "__main__":
    main()
