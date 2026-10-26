"""Make training frames for a part detector, with every box already known.

    python make_detector_data.py                 3,000 training + 400 check frames
    python make_detector_data.py --preview 12    just draw 12, boxes shown, to look at

Why fake frames
---------------
A detector learns from frames where every part has a box drawn round it.
Nobody has drawn boxes on camera frames, and drawing thousands by hand is a
week. But each of the 7,348 dataset photos is one part whose outline we can
measure, so frames can be assembled from them with the boxes known exactly.

Each frame is 1-8 parts on a made-up surface -- paper, wood, a dark desk,
clutter, printed text and rings that are NOT parts -- and then roughed up the
way a camera roughs things up: shadows, glare, tilt, blur, noise, poor JPEG.
Some parts are cut out along their rim, like a real casting on a table; some
keep their grey square, like a print. The detector has to find the part in
both, and ignore everything else.

The split stays clean: training frames are built only from data_grouped\\train
photos and check frames only from data_grouped\\val, so the check score is
measured on parts the detector never saw.

The box
-------
Each box hugs the part's outer rim. Measured on the dataset, the rim's radius
is 1.80 times the dark cavity's (5th-95th percentile 1.77-1.83), so the rim is
found from the cavity, which multi_inspect.py already locates reliably. The
classifiers want the framing of the original photos, which is 1.236 times the
rim's width; that number is written to detect_data\\meta.json for the scanner.

Output (YOLO format, read by train_detector.py)
------
    detect_data\\images\\train\\*.jpg     detect_data\\labels\\train\\*.txt
    detect_data\\images\\val\\*.jpg       detect_data\\labels\\val\\*.txt
    detect_data\\data.yaml                detect_data\\meta.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multi_inspect as M  # noqa: E402  (finds the cavity in each photo)
from live_inspect import cv2, IMAGE_SUFFIXES  # noqa: E402

RIM_PER_CAVITY = 1.80
CROP_PER_RIM_DIAMETER = M.CROP_PER_RADIUS / (2 * RIM_PER_CAVITY)   # 1.236

WORDS = ["QC", "LOT 42", "Batch 7", "PASS", "FAIL", "Station 3", "0.25 mm",
         "ROG", "Villanova", "OK", "B-6", "2026", "Q8", "pump", "08", "60"]


# --------------------------------------------------------------------------
# where each photo's part is
# --------------------------------------------------------------------------

def measure(folder):
    """(path, cx, cy, rim radius) for every photo whose part can be found."""
    paths = sorted(p for p in Path(folder).rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    out, skipped = [], 0
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        parts = M.find_parts(img) if img is not None else []
        if len(parts) != 1:
            skipped += 1
            continue
        q = parts[0]
        out.append((p, q["cx"], q["cy"], RIM_PER_CAVITY * q["r"]))
        if (i + 1) % 500 == 0:
            print(f"\r  measured {i+1}/{len(paths)}", end="", flush=True)
    print(f"\r  {folder}: {len(out)} usable photos, {skipped} skipped "
          f"(part not found cleanly)")
    return out


# --------------------------------------------------------------------------
# surfaces
# --------------------------------------------------------------------------

def background(rng, H, W):
    kind = rng.choice(["paper", "paper", "wood", "desk", "grey", "cloth"])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if kind == "paper":
        base = np.full((H, W, 3), rng.uniform(200, 248), np.float32)
        base *= rng.uniform(0.95, 1.03, 3)                      # slight tint
    elif kind == "wood":
        a = rng.uniform(0, np.pi)
        u = xx * np.cos(a) + yy * np.sin(a)
        grain = (np.sin(u / rng.uniform(4, 12) + 3 * np.sin(u / rng.uniform(60, 150)))
                 + 0.5 * np.sin(u / rng.uniform(1.5, 3)))
        tone = rng.uniform(0.6, 1.1)
        base = np.stack([60 + 10 * grain, 110 + 18 * grain, 165 + 22 * grain], -1) * tone
    elif kind == "desk":
        base = np.full((H, W, 3), rng.uniform(25, 70), np.float32)
    elif kind == "cloth":
        c = rng.uniform(40, 200, 3)
        base = np.ones((H, W, 3), np.float32) * c
        base += 6 * np.sin(xx / 2.0)[..., None] * np.sin(yy / 2.0)[..., None]
    else:
        base = np.full((H, W, 3), rng.uniform(110, 190), np.float32)
    # uneven light across the surface
    g = 1 + rng.uniform(-0.25, 0.25) * (xx / W - 0.5) + rng.uniform(-0.25, 0.25) * (yy / H - 0.5)
    base *= g[..., None]
    base += rng.normal(0, rng.uniform(1, 5), base.shape)
    return np.clip(base, 0, 255).astype(np.uint8)


def clutter(img, rng):
    """Things that are not parts: text (including round letters), rings,
    discs, lines, boxes. The scanner's old finder mistook bold O, 0 and 6 for
    parts; the detector learns they are not."""
    H, W = img.shape[:2]
    for _ in range(rng.integers(0, 6)):
        col = tuple(int(v) for v in rng.uniform(0, 255, 3))
        kind = rng.integers(0, 5)
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        if kind == 0:
            cv2.putText(img, str(rng.choice(WORDS)), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        float(rng.uniform(0.8, 4)), col, int(rng.integers(1, 14)), cv2.LINE_AA)
        elif kind == 1:
            r = int(rng.uniform(15, 120))
            cv2.circle(img, (x, y), r, col, int(rng.integers(3, 30)), cv2.LINE_AA)
        elif kind == 2:
            cv2.circle(img, (x, y), int(rng.uniform(10, 90)), col, -1, cv2.LINE_AA)
        elif kind == 3:
            cv2.line(img, (x, y), (int(rng.integers(0, W)), int(rng.integers(0, H))),
                     col, int(rng.integers(1, 8)), cv2.LINE_AA)
        else:
            w, h = int(rng.uniform(20, 300)), int(rng.uniform(20, 200))
            cv2.rectangle(img, (x, y), (x + w, y + h), col, int(rng.choice([-1, 2, 4])))
    return img


# --------------------------------------------------------------------------
# putting parts down
# --------------------------------------------------------------------------

def part_patch(src, rng, rim_px, mode):
    """The photo rotated and scaled so its rim has radius rim_px. Returns the
    patch, an alpha mask, and the rim centre inside the patch."""
    path, cx, cy, R = src
    img = cv2.imread(str(path))
    s = rim_px / R
    ang = rng.uniform(0, 360)
    if mode == "print":
        side = int(round(img.shape[0] * s))
        Mrot = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2),
                                       float(rng.choice([0, 90, 180, 270]) + rng.uniform(-6, 6)), s)
        Mrot[:, 2] += (side / 2 - img.shape[1] / 2, side / 2 - img.shape[0] / 2)
        patch = cv2.warpAffine(img, Mrot, (side, side), borderMode=cv2.BORDER_REPLICATE)
        alpha = np.ones(patch.shape[:2], np.float32)
        c = Mrot @ np.array([cx, cy, 1.0])
        return patch, alpha, (c[0], c[1])
    # cutout: just the casting, its rim as the edge
    side = int(np.ceil(2 * rim_px * 1.08)) + 4
    Mrot = cv2.getRotationMatrix2D((cx, cy), ang, s)
    Mrot[:, 2] += (side / 2 - cx, side / 2 - cy)
    patch = cv2.warpAffine(img, Mrot, (side, side), borderMode=cv2.BORDER_REPLICATE)
    alpha = np.zeros((side, side), np.float32)
    cv2.circle(alpha, (side * 4 // 2, side * 4 // 2), int(rim_px * 1.01 * 4), 1.0, -1,
               cv2.LINE_AA, shift=2)
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.6, rim_px / 120))
    return patch, alpha, (side / 2, side / 2)


def paste(frame, patch, alpha, x0, y0, shadow, rng):
    H, W = frame.shape[:2]
    h, w = patch.shape[:2]
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xb <= xa or yb <= ya:
        return
    p = patch[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32)
    a = alpha[ya - y0:yb - y0, xa - x0:xb - x0][..., None]
    f = frame[ya:yb, xa:xb].astype(np.float32)
    if shadow:   # a soft shadow under a real part
        sh = cv2.GaussianBlur(alpha, (0, 0), max(2, w / 25))
        dx, dy = int(rng.integers(2, 10)), int(rng.integers(2, 10))
        sh = np.roll(np.roll(sh, dy, 0), dx, 1)[ya - y0:yb - y0, xa - x0:xb - x0][..., None]
        f = f * (1 - 0.45 * sh * (1 - a))
    frame[ya:yb, xa:xb] = np.clip(f * (1 - a) + p * a, 0, 255).astype(np.uint8)


def make_frame(sources, rng, H=720, W=1280):
    frame = clutter(background(rng, H, W), rng)
    n_target = int(rng.integers(1, 9))
    short = min(H, W)
    # one size range per frame (one camera height), a little spread within it
    base = rng.uniform(0.07, 0.24) * short
    placed = []                      # (cx, cy, rim radius, keep-out radius)
    tries = 0
    while len(placed) < n_target and tries < 200:
        tries += 1
        rim = base * rng.uniform(0.85, 1.15)
        mode = "print" if rng.random() < 0.4 else "cutout"
        keep = rim * (1.75 if mode == "print" else 1.06)       # a print's square corners reach 1.75 rims out
        cx, cy = rng.uniform(-0.2 * rim, W + 0.2 * rim), rng.uniform(-0.2 * rim, H + 0.2 * rim)
        # most parts fully in view; a few cut by the edge, as happens live
        inside = rim <= cx <= W - rim and rim <= cy <= H - rim
        if not inside and rng.random() > 0.15:
            continue
        if any(np.hypot(cx - px, cy - py) < (keep + pk) * rng.uniform(0.95, 1.1)
               for px, py, _, pk in placed):
            continue
        src = sources[int(rng.integers(len(sources)))]
        patch, alpha, (pcx, pcy) = part_patch(src, rng, rim, mode)
        paste(frame, patch, alpha, int(round(cx - pcx)), int(round(cy - pcy)),
              mode == "cutout" and rng.random() < 0.8, rng)
        placed.append((cx, cy, rim, keep))
    return frame, [(cx, cy, rim) for cx, cy, rim, _ in placed]


# --------------------------------------------------------------------------
# camera trouble
# --------------------------------------------------------------------------

def camera(frame, parts, rng):
    H, W = frame.shape[:2]
    img = frame.astype(np.float32)
    # shadows across part of the scene -- a hand, the phone, its stand
    for _ in range(rng.integers(0, 3)):
        m = np.zeros((H, W), np.float32)
        pts = np.stack([rng.uniform(-0.2, 1.2, 5) * W, rng.uniform(-0.2, 1.2, 5) * H], 1)
        cv2.fillPoly(m, [cv2.convexHull(pts.astype(np.int32))], 1.0)
        m = cv2.GaussianBlur(m, (0, 0), rng.uniform(15, 60))
        img *= (1 - rng.uniform(0.2, 0.55) * m)[..., None]
    # glare
    for _ in range(rng.integers(0, 2)):
        m = np.zeros((H, W), np.float32)
        cv2.ellipse(m, (int(rng.integers(0, W)), int(rng.integers(0, H))),
                    (int(rng.uniform(30, 250)), int(rng.uniform(20, 150))),
                    rng.uniform(0, 180), 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), rng.uniform(10, 40))
        img += rng.uniform(40, 110) * m[..., None]
    # exposure, contrast, colour cast
    img = (img - 128) * rng.uniform(0.7, 1.25) + 128 + rng.uniform(-35, 35)
    img *= rng.uniform(0.92, 1.08, 3)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # tilt
    boxes = []
    if rng.random() < 0.6:
        d = rng.uniform(0, 0.08, (4, 2)) * [W, H]
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        dst = np.float32(src + d * [[1, 1], [-1, 1], [-1, -1], [1, -1]])
        Hm = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, Hm, (W, H), borderMode=cv2.BORDER_REFLECT)
    else:
        Hm = np.eye(3, dtype=np.float32)
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    for cx, cy, rim in parts:
        ring = np.stack([cx + rim * np.cos(t), cy + rim * np.sin(t)], 1).astype(np.float32)
        ring = cv2.perspectiveTransform(ring[None], Hm)[0]
        x0, y0 = ring.min(0)
        x1, y1 = ring.max(0)
        full = (x1 - x0) * (y1 - y0)
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        if (cx1 - cx0) * (cy1 - cy0) / full < 0.5:      # mostly out of view: no label
            continue
        boxes.append((cx0, cy0, cx1, cy1))
    # blur, noise, compression
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 2.0))
    if rng.random() < 0.2:
        k = int(rng.integers(3, 9))
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        Mk = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
        img = cv2.filter2D(img, -1, cv2.warpAffine(kern, Mk, (k, k)))
    img = np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(0, 7), img.shape),
                  0, 255).astype(np.uint8)
    q = int(rng.integers(45, 96))
    img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)
    return img, boxes


# --------------------------------------------------------------------------

def write_split(sources, n, split, out, rng):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    total = 0
    for i in range(n):
        frame, parts = make_frame(sources, rng)
        img, boxes = camera(frame, parts, rng)
        H, W = img.shape[:2]
        name = f"{split}_{i:05d}"
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(out / "labels" / split / f"{name}.txt", "w") as f:
            for x0, y0, x1, y1 in boxes:
                f.write(f"0 {(x0+x1)/2/W:.6f} {(y0+y1)/2/H:.6f} "
                        f"{(x1-x0)/W:.6f} {(y1-y0)/H:.6f}\n")
        total += len(boxes)
        if (i + 1) % 100 == 0:
            print(f"\r  {split}: {i+1}/{n} frames", end="", flush=True)
    print(f"\r  {split}: {n} frames, {total} parts boxed")


def preview(sources, n, rng, path):
    tiles = []
    for _ in range(n):
        frame, parts = make_frame(sources, rng)
        img, boxes = camera(frame, parts, rng)
        for x0, y0, x1, y1 in boxes:
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 3)
        tiles.append(cv2.resize(img, (640, 360)))
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite(str(path), np.vstack(rows))
    print(f"Preview with boxes: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_grouped")
    ap.add_argument("--out", default="detect_data")
    ap.add_argument("--train-frames", type=int, default=3000)
    ap.add_argument("--val-frames", type=int, default=400)
    ap.add_argument("--preview", type=int, default=0, metavar="N",
                    help="only draw N frames with their boxes, to check by eye")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("Measuring where the part sits in each photo ...")
    train_src = measure(Path(args.data) / "train")
    if args.preview:
        preview(train_src, args.preview, rng, out / "preview.jpg")
        return 0
    val_src = measure(Path(args.data) / "val")
    if len(train_src) < 50 or len(val_src) < 20:
        raise SystemExit("Too few usable photos -- is --data right?")

    write_split(train_src, args.train_frames, "train", out, rng)
    write_split(val_src, args.val_frames, "val", out, rng)

    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: impeller\n", encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({
        "box": "outer rim of the casting",
        "rim_per_cavity_radius": RIM_PER_CAVITY,
        "crop_per_rim_diameter": round(CROP_PER_RIM_DIAMETER, 4),
        "train_frames": args.train_frames, "val_frames": args.val_frames,
        "train_photos": len(train_src), "val_photos": len(val_src),
        "seed": args.seed,
    }, indent=1), encoding="utf-8")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min -> {out}")
    print("Next:  python train_detector.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
