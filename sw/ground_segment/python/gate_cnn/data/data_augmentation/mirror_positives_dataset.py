import json
import sys
from pathlib import Path
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# =========================
# CONFIG
# =========================

# Keys voor de bestandsnaam
POSSIBLE_FILENAME_KEYS = [
    "img_name",
    "image",
    "image_name",
    "filename",
    "file_name",
    "img",
    "path",
]

# Keys voor de heading angle (wordt gespiegeld bij flip)
POSSIBLE_HEADING_KEYS = [
    "heading_angle",
    "heading",
    "angle",
    "yaw_error",
    "steering",
]

# Keys voor confidence (confidence > 0 betekent gate aanwezig)
POSSIBLE_CONFIDENCE_KEYS = [
    "confidence",
    "conf",
    "score",
]

# Keys voor expliciete binaire label (fallback als er geen confidence is)
POSSIBLE_LABEL_KEYS = [
    "label",
    "has_gate",
    "gate_present",
    "gate",
    "target",
    "class",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================
# HELPERS
# =========================

def choose_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Selecteer datasetfolder met images/ en labels")
    return Path(folder) if folder else None


def find_labels_file(dataset_dir: Path) -> Path:
    candidates = [
        dataset_dir / "labels.json",
        dataset_dir / "labels",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c

    json_files = list(dataset_dir.glob("*.json"))
    if len(json_files) == 1:
        return json_files[0]

    raise FileNotFoundError(f"Kon geen labels file vinden in {dataset_dir}")


def find_images_dir(dataset_dir: Path) -> Path:
    candidates = [
        dataset_dir / "images",
        dataset_dir / "image",
        dataset_dir / "imgs",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c

    raise FileNotFoundError(f"Kon geen images-map vinden in {dataset_dir}")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def detect_key(sample: dict, candidates: list):
    for key in candidates:
        if key in sample:
            return key
    return None


def is_positive(sample: dict, confidence_key, label_key) -> bool:
    """
    Bepaalt of een sample een gate bevat:
    - Als confidence aanwezig is: confidence > 0
    - Anders: label == 1 / True / "1"
    """
    if confidence_key and confidence_key in sample:
        try:
            return float(sample[confidence_key]) > 0
        except (ValueError, TypeError):
            return False

    if label_key and label_key in sample:
        val = sample[label_key]
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return val == 1
        if isinstance(val, str):
            return val.strip().lower() in {"1", "true", "yes"}

    return False


def mirror_image(src: Path, dst: Path):
    with Image.open(src) as img:
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped.save(dst)


def make_flipped_name(filename: str) -> str:
    p = Path(filename)
    return f"{p.stem}_flip{p.suffix}"


def already_flipped(filename: str) -> bool:
    return Path(filename).stem.endswith("_flip")


# =========================
# PROCESSORS
# =========================

def process_list_format(labels_data: list, images_dir: Path) -> list:
    if not labels_data:
        return labels_data

    if not isinstance(labels_data[0], dict):
        raise ValueError("Labels list-format bevat geen dictionaries.")

    sample0 = labels_data[0]
    filename_key   = detect_key(sample0, POSSIBLE_FILENAME_KEYS)
    heading_key    = detect_key(sample0, POSSIBLE_HEADING_KEYS)
    confidence_key = detect_key(sample0, POSSIBLE_CONFIDENCE_KEYS)
    label_key      = detect_key(sample0, POSSIBLE_LABEL_KEYS)

    if filename_key is None:
        raise ValueError(
            f"Kon geen filename-key vinden in labels. "
            f"Gevonden keys: {list(sample0.keys())}. "
            f"Voeg de juiste key toe aan POSSIBLE_FILENAME_KEYS."
        )

    if confidence_key is None and label_key is None:
        raise ValueError(
            f"Kon geen label/confidence key vinden in labels. "
            f"Gevonden keys: {list(sample0.keys())}."
        )

    # Bestaande flip-namen bijhouden om duplicaten te vermijden
    existing_flipped = {
        s[filename_key]
        for s in labels_data
        if already_flipped(s[filename_key])
    }

    new_entries = []

    for sample in labels_data:
        filename = sample[filename_key]

        if already_flipped(filename):
            continue

        if not is_positive(sample, confidence_key, label_key):
            continue

        flipped_name = make_flipped_name(Path(filename).name)

        if flipped_name in existing_flipped:
            print(f"Overgeslagen (bestaat al): {flipped_name}")
            continue

        src_img = images_dir / filename
        if not src_img.exists():
            print(f"Waarschuwing: image niet gevonden: {src_img}")
            continue

        dst_flipped = images_dir / flipped_name
        if not dst_flipped.exists():
            mirror_image(src_img, dst_flipped)

        flipped_sample = sample.copy()
        flipped_sample[filename_key] = flipped_name

        if heading_key and flipped_sample.get(heading_key) is not None:
            try:
                flipped_sample[heading_key] = -float(flipped_sample[heading_key])
            except (ValueError, TypeError):
                print(f"Waarschuwing: kon heading niet inverteren voor {filename}")

        new_entries.append(flipped_sample)

    return labels_data + new_entries


def process_dict_format(labels_data: dict, images_dir: Path) -> dict:
    if not labels_data:
        return labels_data

    first_key    = next(iter(labels_data.keys()))
    first_sample = labels_data[first_key]

    if not isinstance(first_sample, dict):
        raise ValueError("Dict-format labels heeft geen dicts als values.")

    heading_key    = detect_key(first_sample, POSSIBLE_HEADING_KEYS)
    confidence_key = detect_key(first_sample, POSSIBLE_CONFIDENCE_KEYS)
    label_key      = detect_key(first_sample, POSSIBLE_LABEL_KEYS)

    if confidence_key is None and label_key is None:
        raise ValueError(
            f"Kon geen label/confidence key vinden in labels. "
            f"Gevonden keys: {list(first_sample.keys())}."
        )

    new_entries = {}

    for filename, sample in labels_data.items():
        if already_flipped(filename):
            continue

        if not is_positive(sample, confidence_key, label_key):
            continue

        flipped_name = make_flipped_name(Path(filename).name)

        if flipped_name in labels_data or flipped_name in new_entries:
            print(f"Overgeslagen (bestaat al): {flipped_name}")
            continue

        src_img = images_dir / filename
        if not src_img.exists():
            print(f"Waarschuwing: image niet gevonden: {src_img}")
            continue

        dst_flipped = images_dir / flipped_name
        if not dst_flipped.exists():
            mirror_image(src_img, dst_flipped)

        flipped_sample = sample.copy()
        if heading_key and flipped_sample.get(heading_key) is not None:
            try:
                flipped_sample[heading_key] = -float(flipped_sample[heading_key])
            except (ValueError, TypeError):
                print(f"Waarschuwing: kon heading niet inverteren voor {filename}")

        new_entries[flipped_name] = flipped_sample

    return {**labels_data, **new_entries}


# =========================
# MAIN
# =========================

def main():
    if len(sys.argv) > 1:
        dataset_dir = Path(sys.argv[1])
        if not dataset_dir.exists():
            print(f"Fout: folder bestaat niet: {dataset_dir}")
            return
    else:
        print("Selecteer de datasetfolder met images/ en labels ...")
        dataset_dir = choose_folder()
        if dataset_dir is None:
            print("Geen folder geselecteerd. Stop.")
            return

    labels_file = find_labels_file(dataset_dir)
    images_dir  = find_images_dir(dataset_dir)

    print(f"Dataset folder : {dataset_dir}")
    print(f"Images folder  : {images_dir}")
    print(f"Labels file    : {labels_file}")

    labels_data = load_json(labels_file)

    if isinstance(labels_data, list):
        new_labels = process_list_format(labels_data, images_dir)
    elif isinstance(labels_data, dict):
        new_labels = process_dict_format(labels_data, images_dir)
    else:
        raise ValueError("Onbekend labels formaat. Verwacht list of dict.")

    save_json(new_labels, labels_file)

    original_count = len(labels_data)
    new_count      = len(new_labels) - original_count

    print("\nKlaar.")
    print(f"Originele entries             : {original_count}")
    print(f"Nieuwe flip entries toegevoegd: {new_count}")
    print(f"Totaal entries                : {len(new_labels)}")


if __name__ == "__main__":
    main()
