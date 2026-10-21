"""Classify castings live from a camera, with the models you already trained.

Nothing here trains. It loads the saved weights once and only predicts, so a
frame takes milliseconds.

    pip install opencv-python                    once
    python live_inspect.py --list-cameras        which number is the iPhone?
    python live_inspect.py --camera 1            live from that camera
    python live_inspect.py --images data_grouped\\val
                                                 fake camera: steps through a folder
    python live_inspect.py --check data_grouped\\val
                                                 proves this script gives the same
                                                 answers as training did

The two models
--------------
stage one   results\\<newest alexnet run>\\model_parameters.pth   good or defective
stage two   defect_types\\stage_two\\model_nopw.pth               fray / chip / spot / slash

Stage two's weights only exist after apply_defect_types.py has been run once
with the version that saves them. Without them this still runs, stage one only.

Each frame goes through exactly what the validation images went through in
training -- greyscale, 224 x 224, no normalisation -- after a centred square
crop. Stage two is only reported when stage one says defective, the same
order as the pipeline.

What a camera changes
---------------------
The models learned from one dataset: top-down photos of one impeller under one
lighting setup. A camera at a different height, angle or light is a different
picture, and accuracy will drop by an amount nobody has measured yet. Line the
part's outer rim up with the guide circle, light it evenly, shoot straight
down. Press S to save frames -- labelled later, those are the training images
that would close the gap.

Keys
----
    q / Esc   quit
    space     pause
    s         save this crop (to captures\\, with the predictions logged)
    [  ]      shrink / grow the crop square
    n  p      next / previous image          (--images only)
    a         autoplay on / off              (--images only)
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# Without this, Windows' Media Foundation backend can take many seconds to
# open a camera. Harmless everywhere else.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

try:
    import cv2
except ImportError:
    raise SystemExit("OpenCV is not installed. Run:  pip install opencv-python")

FLAGS = ["fray", "chip", "spot", "slash"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# BGR, for OpenCV
GREEN = (80, 200, 60)
RED = (60, 60, 235)
AMBER = (40, 170, 240)
WHITE = (235, 235, 235)
GREY = (150, 150, 150)
DARK = (34, 34, 34)

VIEW_HEIGHT = 720
PANEL_WIDTH = 380
GUIDE_RADIUS = 0.41     # the dataset's outer rim spans about 82% of the frame


# --------------------------------------------------------------------------
# finding the models
# --------------------------------------------------------------------------

def norm(p):
    return Path(str(p).replace("\\", "/"))


def tail(p):
    """split/class/filename -- see the filename-collision notes elsewhere."""
    return "/".join(norm(p).parts[-3:])


def newest_stage_one(results_dir="results"):
    runs = [p for p in Path(results_dir).glob("alexnet*")
            if (p / "model_parameters.pth").is_file()]
    if not runs:
        return None
    return max(runs, key=lambda p: (p / "model_parameters.pth").stat().st_mtime)


def stage_one_info(run_dir):
    """Class order and held-out accuracy recorded by that run."""
    names, acc, n = ["def_front", "ok_front"], None, None
    try:
        fm = json.loads((Path(run_dir) / "results.json").read_text())["final_metrics"]
        names = fm.get("class_names") or names
        acc, n = fm.get("accuracy_pct"), fm.get("total_samples")
    except (OSError, ValueError, KeyError):
        pass
    return names, acc, n


def find_stage_two(out_dir="defect_types"):
    """(weights, meta, thresholds) -- the no-pos-weight model if there is one."""
    d = Path(out_dir) / "stage_two"
    for suffix in ("_nopw", ""):
        w = d / f"model{suffix}.pth"
        if w.is_file():
            meta = {}
            try:
                meta = json.loads((d / f"model{suffix}.json").read_text())
            except (OSError, ValueError):
                pass
            return w, meta, d / f"thresholds{suffix}.json"
    return None, {}, d / "thresholds_nopw.json"


def load_cuts(path, cut="f1"):
    cuts = {fl: 0.5 for fl in FLAGS}
    if cut == "half":
        return cuts, "a flat 0.5"
    key = "threshold_f1" if cut == "f1" else "threshold_rate"
    try:
        flags = json.loads(Path(path).read_text())["flags"]
        return ({fl: float(flags[fl].get(key, flags[fl]["threshold"]))
                 for fl in FLAGS}, f"'{cut}' cutoffs from {path}")
    except (OSError, ValueError, KeyError):
        return cuts, f"0.5 for every flag (could not read {path})"


def load_doubtful(out_dir="defect_types", floor=0.6):
    """Flags whose calls were mostly wrong on the held-out labelling round.
    Shown with a '?' so a live count is not read as more than it is."""
    try:
        flags = json.loads((Path(out_dir) / "holdout_score.json")
                           .read_text())["flags"]
    except (OSError, ValueError, KeyError):
        return {}
    return {fl: d["precision"] for fl, d in flags.items()
            if d.get("precision") is not None and d["precision"] < floor}


# --------------------------------------------------------------------------
# the models
# --------------------------------------------------------------------------

class Inspector:
    """Both trained networks, loaded once. predict() takes RGB arrays."""

    def __init__(self, stage_one, class_names, stage_two=None, meta=None):
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
        from PIL import Image
        self.torch, self.Image = torch, Image
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Rebuilt exactly as ManufacturingNet's Network built it: torchvision
        # AlexNet with its last layer swapped for one output per class. The
        # first conv is untouched because the input has 3 channels.
        one = models.alexnet(weights=None)
        one.classifier[-1] = nn.Linear(4096, len(class_names))
        one.load_state_dict(torch.load(stage_one, map_location="cpu",
                                       weights_only=True))
        self.one = one.to(self.device).eval()
        if "def_front" not in class_names:
            raise SystemExit(f"Stage one's classes are {class_names}; "
                             "expected one called def_front.")
        self.def_index = class_names.index("def_front")

        self.two = None
        if stage_two:
            arch = (meta or {}).get("arch", "alexnet")
            if arch == "resnet18":
                two = models.resnet18(weights=None)
                two.fc = nn.Linear(two.fc.in_features, len(FLAGS))
            else:
                two = models.alexnet(weights=None)
                two.classifier[-1] = nn.Linear(4096, len(FLAGS))
            two.load_state_dict(torch.load(stage_two, map_location="cpu",
                                           weights_only=True))
            self.two = two.to(self.device).eval()

        # The validation transform both stages trained against.
        self.tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    def predict(self, rgb_list):
        """-> (p_defective per image, 4 defect scores per image or None)"""
        torch = self.torch
        x = torch.stack([self.tf(self.Image.fromarray(a)) for a in rgb_list])
        x = x.to(self.device)
        with torch.no_grad():
            p1 = torch.softmax(self.one(x), 1)[:, self.def_index].cpu().numpy()
            p2 = (torch.sigmoid(self.two(x)).cpu().numpy()
                  if self.two is not None else None)
        return p1, p2


# --------------------------------------------------------------------------
# per-frame logic -- no torch, so it can be tested without a GPU
# --------------------------------------------------------------------------

def centre_square(frame, frac=1.0):
    """The centred square the models see, and where it sits in the frame."""
    h, w = frame.shape[:2]
    side = max(16, int(min(h, w) * min(max(frac, 0.2), 1.0)))
    y0, x0 = (h - side) // 2, (w - side) // 2
    return frame[y0:y0 + side, x0:x0 + side], (x0, y0, side)


class Smoother:
    """Running average over frames, so the verdict does not flicker as the
    camera's exposure and noise move a few percent each frame."""

    def __init__(self, alpha=0.35):
        self.alpha, self.p_def, self.p_types = alpha, None, None

    def reset(self):
        self.p_def = self.p_types = None

    def update(self, p_def, p_types):
        a = self.alpha
        if self.p_def is None:
            self.p_def = float(p_def)
            self.p_types = None if p_types is None else np.asarray(p_types, float)
        else:
            self.p_def = a * float(p_def) + (1 - a) * self.p_def
            if p_types is not None:
                pt = np.asarray(p_types, float)
                self.p_types = pt if self.p_types is None else a * pt + (1 - a) * self.p_types
        return self.p_def, self.p_types


def decide(p_def, p_types, cuts, threshold=0.5):
    defective = p_def >= threshold
    types = None if p_types is None else dict(zip(FLAGS, map(float, p_types)))
    found = ([fl for fl in FLAGS if types[fl] >= cuts[fl]]
             if defective and types else [])
    return {"defective": bool(defective),
            "confidence": float(p_def if defective else 1 - p_def),
            "p_def": float(p_def), "types": types, "found": found}


def true_label(path):
    parts = {p.lower() for p in norm(path).parts}
    if "def_front" in parts:
        return "defective"
    if "ok_front" in parts:
        return "good"
    return None


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def _text(img, s, xy, scale=0.55, colour=WHITE, thick=1):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick,
                cv2.LINE_AA)


def render(frame, box, result, model_view, info, cuts, doubtful):
    """The camera view with the crop box and guide, plus a results panel."""
    scale = VIEW_HEIGHT / frame.shape[0]
    view = cv2.resize(frame, (max(1, int(frame.shape[1] * scale)), VIEW_HEIGHT),
                      interpolation=cv2.INTER_LINEAR)
    x0, y0, side = (int(v * scale) for v in box)
    colour = WHITE if result is None else (RED if result["defective"] else GREEN)
    cv2.rectangle(view, (x0, y0), (x0 + side, y0 + side), colour, 2)
    cx, cy = x0 + side // 2, y0 + side // 2
    cv2.circle(view, (cx, cy), int(side * GUIDE_RADIUS), GREY, 1, cv2.LINE_AA)
    _text(view, "line the rim up with the circle", (x0 + 8, y0 + side - 10),
          0.5, GREY)

    panel = np.full((VIEW_HEIGHT, PANEL_WIDTH, 3), DARK, np.uint8)
    y = 58
    if result is None:
        _text(panel, "waiting for a frame", (20, y), 0.8, GREY, 2)
    else:
        word = "DEFECTIVE" if result["defective"] else "GOOD"
        _text(panel, word, (20, y), 1.5, colour, 3)
        y += 34
        _text(panel, f"confidence {100*result['confidence']:.1f}%", (22, y), 0.6, WHITE)
        y += 44
        if result["defective"] and result["types"]:
            _text(panel, "Defect type", (20, y), 0.65, WHITE, 2)
            y += 26
            for fl in FLAGS:
                p, hit = result["types"][fl], fl in result["found"]
                doubt = fl in doubtful
                label = fl + ("?" if doubt else "")
                c = (AMBER if doubt else RED) if hit else GREY
                _text(panel, label, (22, y + 14), 0.6, c, 2 if hit else 1)
                bx, bw = 110, 190
                cv2.rectangle(panel, (bx, y), (bx + bw, y + 18), (70, 70, 70), -1)
                cv2.rectangle(panel, (bx, y), (bx + int(bw * p), y + 18), c, -1)
                tick = bx + int(bw * cuts[fl])
                cv2.line(panel, (tick, y - 3), (tick, y + 21), WHITE, 1)
                _text(panel, f"{p:.2f}", (bx + bw + 10, y + 14), 0.5, WHITE)
                y += 32
            if not result["found"]:
                _text(panel, "none of the four over its cutoff", (22, y + 4), 0.5, GREY)
                y += 22
            if doubtful:
                y += 6
                for fl, prec in doubtful.items():
                    _text(panel, f"{fl}? : {100*prec:.0f}% of its calls were real",
                          (22, y), 0.45, AMBER)
                    y += 18
                    _text(panel, "      on the held-out test -- a hint, not a call",
                          (22, y), 0.45, AMBER)
                    y += 20
            _text(panel, "white tick = cutoff", (22, y + 6), 0.45, GREY)
        elif result["defective"]:
            _text(panel, "defect-type model not loaded", (22, y), 0.5, GREY)
        else:
            _text(panel, "no defect-type check needed", (22, y), 0.5, GREY)

    if model_view is not None:
        size = 150
        small = cv2.resize(model_view, (size, size), interpolation=cv2.INTER_AREA)
        if small.ndim == 2:
            small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        top = VIEW_HEIGHT - size - 120
        panel[top:top + size, 20:20 + size] = small
        _text(panel, "what the model sees", (20, top - 8), 0.45, GREY)

    yb = VIEW_HEIGHT - 92
    if info.get("truth"):
        ok = result is not None and (result["defective"] == (info["truth"] == "defective"))
        _text(panel, f"file says {info['truth']}: " + ("match" if ok else "MISMATCH"),
              (20, yb), 0.5, GREEN if ok else RED)
    yb += 20
    _text(panel, info.get("source", ""), (20, yb), 0.45, GREY)
    yb += 18
    status = [f"{info['fps']:.0f} fps"] if info.get("fps") else []
    if info.get("paused"):
        status.append("PAUSED")
    if info.get("saved"):
        status.append(f"{info['saved']} saved")
    _text(panel, "   ".join(status), (20, yb), 0.45, GREY)
    yb += 18
    _text(panel, info.get("keys", ""), (20, yb), 0.42, GREY)
    return np.hstack([view, panel])


# --------------------------------------------------------------------------
# saving frames for future labelling
# --------------------------------------------------------------------------

def save_capture(crop, result, source, folder="captures"):
    d = Path(folder)
    d.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time()*1000)%1000:03d}"
    path = d / f"capture-{stamp}.png"
    cv2.imwrite(str(path), crop)
    log = d / "log.csv"
    new = not log.is_file()
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "time", "source", "verdict", "p_defective"]
                       + [f"p_{fl}" for fl in FLAGS])
        types = (result or {}).get("types") or {}
        w.writerow([path.name, stamp, source,
                    "" if result is None else ("defective" if result["defective"] else "good"),
                    "" if result is None else f"{result['p_def']:.4f}"]
                   + [f"{types[fl]:.4f}" if fl in types else "" for fl in FLAGS])
    return path


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------

BACKENDS = {"dshow": "CAP_DSHOW", "msmf": "CAP_MSMF", "any": "CAP_ANY"}


def backend_order(choice="auto"):
    """Windows has two camera systems. Older DirectShow is what most webcams
    answer on; newer Media Foundation is how some virtual cameras -- Camo
    without its compatibility add-on, for one -- publish their picture, and
    those show up as a black frame on DirectShow. 'auto' tries both."""
    if choice != "auto":
        return [choice]
    return ["dshow", "msmf"] if os.name == "nt" else ["any"]


def _mute_opencv():
    # Probing cameras that are absent or black makes OpenCV print a wall of
    # warnings; they are expected while searching, so mute them.
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except AttributeError:
        pass


def try_open(index, name, width=1280, height=720):
    """(capture or None, first frame, still black?) on one backend."""
    cap = cv2.VideoCapture(index, getattr(cv2, BACKENDS[name], cv2.CAP_ANY))
    if not cap.isOpened():
        cap.release()
        return None, None, True
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frame, dark = warm_up(cap)
    if frame is None:
        cap.release()
        return None, None, True
    return cap, frame, dark


def open_camera(index, choice="auto"):
    """The first backend that gives a real picture. If every one that opens
    stays black, fall back to the first of those, so the window still shows
    what is (not) arriving."""
    _mute_opencv()
    fallback = None
    for name in backend_order(choice):
        cap, frame, dark = try_open(index, name)
        if cap is None:
            continue
        if not dark:
            return cap, name, False
        cap.release()
        fallback = fallback or name
    if fallback:
        cap, _, _ = try_open(index, fallback)
        if cap is not None:
            return cap, fallback, True
    return None, None, True


def list_cameras(n=6, folder="captures", choice="auto"):
    _mute_opencv()
    Path(folder).mkdir(exist_ok=True)
    found, black = 0, []
    for i in range(n):
        best = None
        for name in backend_order(choice):
            cap, frame, dark = try_open(i, name)
            if cap is None:
                continue
            cap.release()
            print(f"  camera {i} ({name:<5}) {frame.shape[1]} x {frame.shape[0]}   "
                  + ("BLACK -- no picture" if dark else "picture"))
            if best is None or (best[1] and not dark):
                best = (frame, dark)
        if best is None:
            continue
        found += 1
        snap = Path(folder) / f"camera-{i}.png"
        cv2.imwrite(str(snap), best[0])
        print(f"             snapshot -> {snap}")
        if best[1]:
            black.append(i)
    if not found:
        print("  No cameras answered. With the iPhone: is Camo open on both "
              "the phone and the PC?")
        return
    if black:
        print(f"\nStill black on every backend: camera {', '.join(map(str, black))}."
              "\nIf that is Camo and Camo Studio itself shows the phone, install"
              "\nCamo's compatibility add-on: click the notifications icon at the"
              "\ntop right of Camo Studio. Then close Camo Studio's other uses of"
              "\nthe camera, or any other app using one, and run this again.")
    print("\nOpen the snapshots to see which number is the iPhone, then:"
          "\n  python live_inspect.py --camera <that number>")


def warm_up(cap, seconds=3.0):
    """A camera's first frames are often black while it starts and sets its
    exposure. Read for a few seconds until one has a picture in it."""
    end, frame = time.time() + seconds, None
    while time.time() < end:
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
            if f.max() > 16:
                return f, False
    return frame, True


def image_files(folder):
    return sorted(p for p in Path(folder).rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def read_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------
# --check: same answers as training?
# --------------------------------------------------------------------------

def source_of(name):
    m = re.match(r"^from-(train|val)__(.+)$", name)
    return (m.group(1), m.group(2)) if m else (None, name)


def stage_two_reference(predicted_csv):
    """predicted.csv is keyed on data_clean paths; data_grouped renames only
    the colliding filenames, with the source split as a prefix."""
    by_tail, by_name = {}, {}
    try:
        with open(predicted_csv, newline="") as f:
            for r in csv.DictReader(f):
                p = {fl: float(r[f"p_{fl}"]) for fl in FLAGS}
                by_tail[tail(r["file"])] = p
                by_name.setdefault(norm(r["file"]).name, []).append(p)
    except OSError:
        return None

    def lookup(path):
        split, name = source_of(norm(path).name)
        if split:
            return by_tail.get(f"{split}/def_front/{name}")
        hits = by_name.get(name, [])
        return hits[0] if len(hits) == 1 else None
    return lookup


def compare(files, p1, p2, s1_ref, s2_lookup, cuts, threshold=0.5):
    """Pure bookkeeping, separated so it can be tested without the models."""
    out = {"files": len(files), "s1_compared": 0, "s1_same": 0, "s1_maxdiff": 0.0,
           "truth_n": 0, "truth_right": 0,
           "s2_compared": 0, "s2_maxdiff": 0.0, "s2_flag_same": 0, "s2_flag_n": 0}
    for i, path in enumerate(files):
        pd = float(p1[i])
        t = true_label(path)
        if t:
            out["truth_n"] += 1
            out["truth_right"] += (pd >= threshold) == (t == "defective")
        ref = s1_ref.get(tail(path))
        if ref:
            out["s1_compared"] += 1
            ref_def = ref["predicted"] == "def_front"
            out["s1_same"] += (pd >= threshold) == ref_def
            ref_pdef = ref["confidence"] if ref_def else 1 - ref["confidence"]
            out["s1_maxdiff"] = max(out["s1_maxdiff"], abs(pd - ref_pdef))
        if p2 is not None and s2_lookup is not None and t == "defective":
            r2 = s2_lookup(path)
            if r2:
                out["s2_compared"] += 1
                for k, fl in enumerate(FLAGS):
                    d = abs(float(p2[i][k]) - r2[fl])
                    out["s2_maxdiff"] = max(out["s2_maxdiff"], d)
                    out["s2_flag_n"] += 1
                    out["s2_flag_same"] += ((p2[i][k] >= cuts[fl]) == (r2[fl] >= cuts[fl]))
    return out


def run_check(inspector, folder, run_dir, predicted_csv, cuts, batch=32):
    files = image_files(folder)
    if not files:
        raise SystemExit(f"No images under {folder}")
    s1_ref = {}
    try:
        with open(Path(run_dir) / "predictions.csv", newline="") as f:
            for r in csv.DictReader(f):
                s1_ref[tail(r["file"])] = {"predicted": r["predicted"],
                                           "confidence": float(r["confidence"])}
    except OSError:
        pass
    s2_lookup = stage_two_reference(predicted_csv) if inspector.two else None

    print(f"Checking {len(files)} images in {folder} ...")
    P1, P2, kept = [], [], []
    for i in range(0, len(files), batch):
        chunk, rgbs = [], []
        for p in files[i:i + batch]:
            _, rgb = read_rgb(p)
            if rgb is not None:
                chunk.append(p)
                rgbs.append(rgb)
        if not rgbs:
            continue
        a, b = inspector.predict(rgbs)
        P1.extend(a)
        if b is not None:
            P2.extend(b)
        kept.extend(chunk)
        print(f"\r  {len(kept)}/{len(files)}", end="", flush=True)
    print()
    r = compare(kept, P1, P2 if P2 else None, s1_ref, s2_lookup, cuts)

    print()
    print("=" * 66)
    print("  DOES THE LIVE PATH GIVE TRAINING'S ANSWERS?")
    print("=" * 66)
    if r["truth_n"]:
        print(f"  stage one accuracy here   {r['truth_right']}/{r['truth_n']}"
              f"  ({100*r['truth_right']/r['truth_n']:.2f}%)")
    if r["s1_compared"]:
        print(f"  stage one vs training     {r['s1_same']}/{r['s1_compared']} "
              f"same verdict, largest gap {r['s1_maxdiff']:.4f}")
    else:
        print(f"  stage one vs training     no predictions.csv in {run_dir} to compare")
    if r["s2_compared"]:
        print(f"  stage two vs predicted.csv {r['s2_flag_same']}/{r['s2_flag_n']} "
              f"same flag calls on {r['s2_compared']} castings, largest gap "
              f"{r['s2_maxdiff']:.4f}")
    elif inspector.two is not None:
        print("  stage two vs predicted.csv nothing to compare (no matching rows)")
    ok1 = not r["s1_compared"] or (r["s1_same"] == r["s1_compared"]
                                   and r["s1_maxdiff"] < 0.02)
    ok2 = not r["s2_compared"] or r["s2_maxdiff"] < 0.02
    print()
    if ok1 and ok2:
        print("  PASS -- the live script reproduces training's predictions, so")
        print("  any difference you see on a camera comes from the picture,")
        print("  not from this code.")
    else:
        print("  MISMATCH -- the live preprocessing or weights differ from what")
        print("  produced those files. If you retrained one stage after making")
        print("  them, regenerate the file and check again.")
    print("=" * 66)
    return 0 if ok1 and ok2 else 1


# --------------------------------------------------------------------------
# the live loop
# --------------------------------------------------------------------------

def run_live(inspector, args, cuts, doubtful):
    win = "Casting inspection"
    folder_mode = args.images is not None
    if folder_mode:
        files = image_files(args.images)
        if not files:
            raise SystemExit(f"No images under {args.images}")
        idx, autoplay, last_step = 0, False, 0.0
        keys = "q quit  space pause  s save  n/p next/prev  a auto"
    else:
        print(f"Opening camera {args.camera} ...")
        cap, used, dark = open_camera(args.camera, args.backend)
        if cap is None:
            raise SystemExit(f"Camera {args.camera} did not open. "
                             "Try:  python live_inspect.py --list-cameras")
        print(f"Camera {args.camera} via {used}"
              + ("  -- BLACK on every backend; see --list-cameras for the fix"
                 if dark else ""))
        keys = "q quit  space pause  s save  [ ] crop size"

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    smooth = Smoother(args.smooth)
    frac, paused, saved = args.crop, False, 0
    frame = result = crop = None
    t_prev, fps = time.time(), 0.0
    loaded_idx = -1

    while True:
        truth, source = None, f"camera {args.camera}"
        if folder_mode:
            if autoplay and not paused and time.time() - last_step > 1.2:
                idx, last_step = (idx + 1) % len(files), time.time()
            if idx != loaded_idx:
                frame, _ = read_rgb(files[idx])
                loaded_idx, result = idx, None
                smooth.reset()
            truth = true_label(files[idx])
            source = f"{idx+1}/{len(files)}  {files[idx].name}"
            fresh = result is None
        else:
            if not paused:
                ok, f = cap.read()
                if not ok:
                    print("Camera stopped sending frames.")
                    break
                frame = f
            fresh = not paused

        if frame is None:
            if folder_mode:
                idx = (idx + 1) % len(files)
                continue
            break

        crop, box = centre_square(frame, 1.0 if folder_mode else frac)
        if fresh:
            p1, p2 = inspector.predict([cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)])
            pd, pt = p1[0], (None if p2 is None else p2[0])
            if not folder_mode:
                pd, pt = smooth.update(pd, pt)
            result = decide(pd, pt, cuts, args.threshold)
        model_view = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (224, 224),
                                interpolation=cv2.INTER_AREA)

        now = time.time()
        if not paused and now > t_prev:
            fps = 0.9 * fps + 0.1 * (1.0 / (now - t_prev)) if fps else 1.0 / (now - t_prev)
        t_prev = now
        canvas = render(frame, box, result, model_view,
                        {"truth": truth, "source": source, "fps": fps,
                         "paused": paused, "saved": saved, "keys": keys},
                        cuts, doubtful)
        cv2.imshow(win, canvas)

        k = cv2.waitKey(30 if folder_mode else 1) & 0xFF
        if k in (ord("q"), 27):
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if k == ord(" "):
            paused = not paused
        elif k == ord("s") and crop is not None:
            path = save_capture(crop, result, source)
            saved += 1
            print(f"saved {path}")
        elif k == ord("[") and not folder_mode:
            frac = max(0.2, frac - 0.05)
            smooth.reset()
        elif k == ord("]") and not folder_mode:
            frac = min(1.0, frac + 0.05)
            smooth.reset()
        elif folder_mode and k == ord("n"):
            idx = (idx + 1) % len(files)
        elif folder_mode and k == ord("p"):
            idx = (idx - 1) % len(files)
        elif folder_mode and k == ord("a"):
            autoplay, last_step = not autoplay, time.time()

    if not folder_mode:
        cap.release()
    cv2.destroyAllWindows()
    return 0


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0,
                     help="camera number (see --list-cameras)")
    src.add_argument("--images", metavar="DIR",
                     help="step through a folder of images instead of a camera")
    src.add_argument("--check", metavar="DIR",
                     help="run every image in DIR and compare with training's "
                          "saved predictions; no window")
    src.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--backend", choices=["auto", "dshow", "msmf", "any"],
                    default="auto",
                    help="Windows camera system; 'auto' tries DirectShow, then "
                         "Media Foundation if the picture is black")
    ap.add_argument("--stage-one", metavar="PTH",
                    help="stage one weights (default: newest results\\alexnet* run)")
    ap.add_argument("--stage-two", metavar="PTH",
                    help="stage two weights (default: defect_types\\stage_two\\"
                         "model_nopw.pth)")
    ap.add_argument("--thresholds", metavar="JSON",
                    help="stage two cutoffs (default: the file matching the model)")
    ap.add_argument("--cut", choices=["f1", "rate", "half"], default="f1",
                    help="'f1' is the per-casting cutoff -- the right one for "
                         "calling a single part")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="stage one: defective when its probability is at least this")
    ap.add_argument("--crop", type=float, default=0.9,
                    help="crop square as a fraction of the frame's short side")
    ap.add_argument("--smooth", type=float, default=0.35,
                    help="weight of the newest frame in the running average "
                         "(1 = no smoothing)")
    ap.add_argument("--predicted", default="defect_types/predicted.csv",
                    help="--check compares stage two against this")
    args = ap.parse_args()

    if args.list_cameras:
        list_cameras(choice=args.backend)
        return 0

    if args.stage_one:
        s1, run_dir = Path(args.stage_one), Path(args.stage_one).parent
    else:
        run_dir = newest_stage_one()
        if run_dir is None:
            raise SystemExit("No results\\alexnet*\\model_parameters.pth found. "
                             "Train stage one first:  python run.py --data "
                             "data_grouped augment < answers_es.txt")
        s1 = run_dir / "model_parameters.pth"
    names, acc, n = stage_one_info(run_dir)

    if args.stage_two:
        s2, meta = Path(args.stage_two), {}
        thr = Path(args.thresholds) if args.thresholds else (
            s2.parent / ("thresholds_nopw.json" if "nopw" in s2.name
                         else "thresholds.json"))
    else:
        s2, meta, thr = find_stage_two()
        if args.thresholds:
            thr = Path(args.thresholds)
    cuts, cut_src = load_cuts(thr, args.cut)
    doubtful = load_doubtful()

    print(f"Stage one : {s1}")
    if acc is not None:
        print(f"            {acc}% on the {n} held-out castings when it was trained")
    if s2:
        print(f"Stage two : {s2}")
        print(f"            yes/no from {cut_src}")
    else:
        print("Stage two : not found -- defect types will not be shown.")
        print("            Run once to save it:  python apply_defect_types.py "
              "--no-pos-weight")
    print("Loading ...")
    inspector = Inspector(s1, names, s2, meta)
    print(f"Running on {inspector.device}")

    if args.check:
        return run_check(inspector, args.check, run_dir, args.predicted, cuts)
    return run_live(inspector, args, cuts, doubtful)


if __name__ == "__main__":
    sys.exit(main())
