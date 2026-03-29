import sys
import pillow_heif
from pathlib import Path
from PIL import Image
import tkinter as tk
from tkinter import filedialog

pillow_heif.register_heif_opener()

TARGET_W = 240
TARGET_H = 520
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}


def choose_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Selecteer folder met images om te verwerken")
    return Path(folder) if folder else None


def process_image(img_path: Path):
    with Image.open(img_path) as img:
        # Converteer altijd naar RGB (fix voor HEIC, RGBA, grayscale, etc.)
        img = img.convert("RGB")

        # 1. Roteer 90° rechtsom
        img = img.rotate(-90, expand=True)

        # 2. Schaal zodat de kortste kant past, behoud aspectratio
        w, h = img.size
        scale = max(TARGET_W / w, TARGET_H / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # 3. Crop middenstuk naar exact 240x520
        left = (new_w - TARGET_W) // 2
        top  = (new_h - TARGET_H) // 2
        img  = img.crop((left, top, left + TARGET_W, top + TARGET_H))

        # Sla altijd op als JPEG (ook als het een .heic was)
        output_path = img_path.with_suffix(".jpg")
        img.save(output_path, "JPEG", quality=95)

        # Verwijder origineel als het geen jpg was
        if img_path.suffix.lower() not in {".jpg", ".jpeg"}:
            img_path.unlink()


def main():
    if len(sys.argv) > 1:
        images_dir = Path(sys.argv[1])
        if not images_dir.exists():
            print(f"Fout: folder bestaat niet: {images_dir}")
            return
    else:
        print("Selecteer de folder met images ...")
        images_dir = choose_folder()
        if images_dir is None:
            print("Geen folder geselecteerd. Stop.")
            return

    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        print(f"Geen images gevonden in {images_dir}")
        return

    print(f"Gevonden: {len(image_files)} images")
    print(f"Verwerking: HEIC→JPEG, 90° rechtsom roteren, bijsnijden naar {TARGET_W}x{TARGET_H}")

    for i, img_path in enumerate(image_files, 1):
        process_image(img_path)
        print(f"[{i}/{len(image_files)}] {img_path.name} → {img_path.stem}.jpg")

    print(f"\nKlaar. {len(image_files)} images verwerkt.")


if __name__ == "__main__":
    main()
