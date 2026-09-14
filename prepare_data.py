"""Turn whatever shape your image data arrives in into the layout the models want.

Usage:

    python prepare_data.py "C:\\Users\\jackm\\Downloads\\casting_data.zip"
    python prepare_data.py "C:\\path\\to\\an\\already-extracted\\folder"

What it does:

1. Extracts the archive (if you gave it one) into `raw/`.
2. Walks what's inside and prints the structure it found, with image counts.
3. Works out whether the data already has a train/test split or is one folder
   per class, and builds `data/train` + `data/val` accordingly.
4. Also builds `data_small/` -- a few images per class -- for a fast first run
   that proves the pipeline works before you commit to the full dataset.
5. Prints the exact lines to paste into Python to start training.

It never modifies your original archive, and never deletes anything.

Options:
    --out NAME        output folder for the full dataset  (default: data)
    --small-out NAME  output folder for the subset        (default: data_small)
    --val-split F     fraction held out for validation    (default: 0.2)
    --small N         images per class in the subset      (default: 60)
    --seed N          shuffle seed                        (default: 42)
    --no-small        skip building the subset
"""

import argparse
import random
import shutil
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Directory names that mean "this is a split", not "this is a class".
TRAIN_NAMES = {"train", "training", "train_set"}
VAL_NAMES = {"val", "valid", "validation", "test", "testing", "test_set", "eval"}


def images_in(directory):
    """Image files directly inside `directory` (recursively)."""
    return [p for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]


def subdirs(directory):
    return sorted(d for d in directory.iterdir()
                  if d.is_dir() and not d.name.startswith("."))


def describe(directory, indent=0, max_depth=3):
    """Print a compact tree with image counts, so you can see what you have."""
    if indent > max_depth:
        return
    for child in subdirs(directory):
        count = len(images_in(child))
        pad = "    " * indent
        label = f"{count} images" if count else "-"
        print(f"{pad}  {child.name}/  ({label})")
        if indent < max_depth:
            describe(child, indent + 1, max_depth)


def find_data_root(start):
    """Walk down through single-child wrapper folders to the real root.

    Archives often unpack to casting_data/casting_data/... -- this skips those
    pointless nesting levels so you don't have to.
    """
    current = start
    while True:
        children = subdirs(current)
        direct_images = [p for p in current.iterdir()
                         if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        if len(children) == 1 and not direct_images:
            current = children[0]
            continue
        return current


def classify_layout(root):
    """Return ('split', {'train': dir, 'val': dir}) or ('flat', {class: dir})."""
    children = subdirs(root)
    names = {d.name.lower(): d for d in children}

    train_dir = next((names[n] for n in TRAIN_NAMES if n in names), None)
    val_dir = next((names[n] for n in VAL_NAMES if n in names), None)

    if train_dir is not None:
        return "split", {"train": train_dir, "val": val_dir}

    class_dirs = OrderedDict()
    for child in children:
        if images_in(child):
            class_dirs[child.name] = child
    if class_dirs:
        return "flat", class_dirs

    return "unknown", {}


def copy_split(class_images, out_root, split_name):
    written = 0
    for class_name, images in class_images.items():
        target = out_root / split_name / class_name
        target.mkdir(parents=True, exist_ok=True)
        for image in images:
            shutil.copy2(str(image), str(target / image.name))
            written += 1
    return written


def build_from_flat(class_dirs, out, val_split, seed):
    rng = random.Random(seed)
    train_images, val_images = OrderedDict(), OrderedDict()

    for class_name, directory in class_dirs.items():
        images = sorted(images_in(directory))
        rng.shuffle(images)
        n_val = max(1, int(round(len(images) * val_split)))
        val_images[class_name] = images[:n_val]
        train_images[class_name] = images[n_val:]

    copy_split(train_images, out, "train")
    copy_split(val_images, out, "val")
    return train_images, val_images


def build_from_split(train_dir, val_dir, out, val_split, seed):
    train_classes = OrderedDict(
        (d.name, sorted(images_in(d))) for d in subdirs(train_dir)
    )

    if val_dir is not None:
        val_classes = OrderedDict(
            (d.name, sorted(images_in(d))) for d in subdirs(val_dir)
        )
    else:
        # A train folder with no test folder: carve a validation set out.
        print("  No test/val folder found -- splitting the training set.")
        rng = random.Random(seed)
        val_classes = OrderedDict()
        for class_name, images in train_classes.items():
            images = list(images)
            rng.shuffle(images)
            n_val = max(1, int(round(len(images) * val_split)))
            val_classes[class_name] = images[:n_val]
            train_classes[class_name] = images[n_val:]

    copy_split(train_classes, out, "train")
    copy_split(val_classes, out, "val")
    return train_classes, val_classes


def build_small(train_images, val_images, small_out, per_class, seed):
    rng = random.Random(seed)
    small_train, small_val = OrderedDict(), OrderedDict()
    n_val = max(2, per_class // 4)

    for class_name, images in train_images.items():
        pool = list(images)
        rng.shuffle(pool)
        small_train[class_name] = pool[:per_class]
    for class_name, images in val_images.items():
        pool = list(images)
        rng.shuffle(pool)
        small_val[class_name] = pool[:n_val]

    copy_split(small_train, small_out, "train")
    copy_split(small_val, small_out, "val")
    return small_train, small_val


def report(title, class_images):
    total = sum(len(v) for v in class_images.values())
    print(f"\n  {title}: {total} images")
    for class_name, images in class_images.items():
        share = 100.0 * len(images) / total if total else 0
        print(f"    {class_name:<24} {len(images):>6}  ({share:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="path to a .zip archive or a folder")
    parser.add_argument("--out", default="data")
    parser.add_argument("--small-out", default="data_small")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--small", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-small", action="store_true")
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    if not source.exists():
        print(f"ERROR: nothing at {source}")
        return 1

    # --- 1. get an extracted folder -------------------------------------
    if source.is_file():
        if not zipfile.is_zipfile(source):
            print(f"ERROR: {source.name} is not a zip archive.")
            print("If it came from Google Drive it may be an HTML error page "
                  "renamed to .zip. Download it again in a browser.")
            return 1
        extract_to = Path("raw").resolve()
        print(f"Extracting {source.name} -> {extract_to}")
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            archive.extractall(extract_to)
        root = extract_to
    else:
        root = source.resolve()

    # --- 2. find the real root and show what's there ---------------------
    root = find_data_root(root)
    print(f"\nData root: {root}")
    print("Structure found:")
    describe(root)

    total_images = len(images_in(root))
    print(f"\n  Total images: {total_images}")
    if total_images == 0:
        print("\nERROR: no image files found anywhere under that folder.")
        print("Supported: " + ", ".join(sorted(IMAGE_SUFFIXES)))
        return 1

    # --- 3. work out the layout -----------------------------------------
    layout, found = classify_layout(root)
    out = Path(args.out).resolve()

    if out.exists():
        print(f"\nERROR: {out} already exists. Delete or rename it first, so "
              "an old run can't get mixed in with a new one.")
        return 1

    print(f"\nLayout: {layout}")

    if layout == "split":
        print(f"  train: {found['train'].name}")
        print(f"  val:   {found['val'].name if found['val'] else '(none)'}")
        train_images, val_images = build_from_split(
            found["train"], found["val"], out, args.val_split, args.seed)
    elif layout == "flat":
        print(f"  {len(found)} class folders: {', '.join(found)}")
        train_images, val_images = build_from_flat(
            found, out, args.val_split, args.seed)
    else:
        print("\nERROR: couldn't work out the layout. Expected either one "
              "folder per class, or train/ and test/ folders. Send me the "
              "structure printed above and I'll adjust this script.")
        return 1

    report("Training set", train_images)
    report("Validation set", val_images)

    train_dir = out / "train"
    val_dir = out / "val"

    # --- 4. small subset for a first run ---------------------------------
    small_dir = None
    if not args.no_small:
        small_out = Path(args.small_out).resolve()
        if small_out.exists():
            print(f"\nSkipping subset: {small_out} already exists.")
        else:
            small_train, small_val = build_small(
                train_images, val_images, small_out, args.small, args.seed)
            report(f"Subset for a first run ({args.small_out})", small_train)
            small_dir = small_out

    # --- 5. tell them exactly what to run --------------------------------
    print("\n" + "=" * 62)
    print("DONE. Start with the small subset:")
    print("=" * 62)
    if small_dir:
        print(f"""
from ManufacturingNet.models import AlexNet

AlexNet(r"{small_dir / 'train'}",
        r"{small_dir / 'val'}")
""")
        print("Once that finishes and writes a results.json, run the full set:")
    print(f"""
from ManufacturingNet.models import AlexNet

AlexNet(r"{train_dir}",
        r"{val_dir}")
""")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
