"""Inspect several castings at once, and keep a record of every one.

    python multi_inspect.py --camera 1              live: find every part in view
    python multi_inspect.py --camera 1 --auto       log each part once it settles
    python multi_inspect.py --folder synthetic      every image in a folder
    python multi_inspect.py --selftest data_grouped\\val
                                                    proves finding + classifying
                                                    works, on frames with known
                                                    answers

How it finds the parts
----------------------
The models judge one centred part per picture, so each part has to be found
and cut out first. Two ways, same result shape:

  trained detector   detector\\impeller.pt from train_detector.py, on the GPU.
                     Draws a box round each rim; the crop is 1.236 times the
                     rim's width, the framing of the dataset photos. Used
                     whenever that file exists (--finder auto).
  dark-cavity finder no training: finds the dark round cavity round the hub
                     and crops 4.45 times its radius. Shadows and grey prints
                     can hide parts from it. --finder cavity forces it.

That needs:
    a light, plain background   white paper is ideal; a dark desk is not
    parts not touching          two touching cavities still split, but their
                                crops overlap and each sees part of the other
    parts not too small         every crop is shrunk to 224 x 224 anyway, so
                                a part ~200 pixels across loses little; one 60
                                across loses the small spots

Where the record goes
---------------------
Each run starts a folder, scans\\<date-time>\\, holding

    parts.csv       one row per inspected part
    session.json    the same records in the casting-inspection/1 format that
                    export_results.py writes, with the same summary block --
                    so the front end reads it unchanged. Rewritten after every
                    scan, all at once, so a page polling it never sees half
    crops\\          the exact picture each verdict was made on

A part is only written down when you say so (C, or --auto), not every frame:
the camera sees the same part thirty times a second.

Keys (live)
-----------
    c         scan: record every part in view now
    up/down   step through the parts in view (or the mouse wheel, or click
              a part); the panel shows eight at a time and follows
    f         save the whole camera frame (to see why a part was missed)
    space     pause
    d         show what the finder sees (the dark-cavity mask)
    q / Esc   quit
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_inspect as L  # noqa: E402  (also checks OpenCV is installed)
from live_inspect import cv2, FLAGS  # noqa: E402

# Crop side as a multiple of the dark cavity's radius. Measured on the
# dataset: the cavity's median radius is 67.5 px in the 300 px photos.
CROP_PER_RADIUS = 4.45

PANEL_WIDTH = 400
YELLOW = (0, 230, 255)      # BGR: marks the part selected with the arrows
# arrow keys as cv2.waitKeyEx reports them: Windows, then Linux (GTK)
UP_KEYS = {2490368, 65362}
DOWN_KEYS = {2621440, 65364}
VIEW_HEIGHT = 720


# --------------------------------------------------------------------------
# finding parts
# --------------------------------------------------------------------------

def _shape(c, off=(0, 0)):
    hull = cv2.convexHull(c)
    ha = cv2.contourArea(hull)
    if ha <= 0:
        return None
    per = cv2.arcLength(hull, True)
    (_, _), rr = cv2.minEnclosingCircle(c)
    m = cv2.moments(hull)
    return {"cx": m["m10"] / m["m00"] + off[0], "cy": m["m01"] / m["m00"] + off[1],
            "r": float(np.sqrt(ha / np.pi)),
            "circ": 4 * np.pi * ha / (per * per + 1e-9),
            "fill": ha / (np.pi * rr * rr + 1e-9),
            "solid": cv2.contourArea(c) / ha}


def _round_enough(s):
    return (s is not None and s["circ"] > 0.8 and s["fill"] > 0.75
            and s["solid"] > 0.8)


def _ring_mean(g, cx, cy, r0, r1):
    h, w = g.shape
    x0, x1 = int(max(0, cx - r1)), int(min(w, cx + r1 + 1))
    y0, y1 = int(max(0, cy - r1)), int(min(h, cy + r1 + 1))
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - cx, yy - cy)
    sel = (d >= r0) & (d <= r1)
    return float(g[y0:y1, x0:x1][sel].mean()) if sel.any() else None


def _looks_like_impeller(b, s):
    """Not just any dark round thing: a brighter hub in the middle of the
    cavity and a brighter face around it."""
    cav = _ring_mean(b, s["cx"], s["cy"], 0.62 * s["r"], 0.92 * s["r"])
    hub = _ring_mean(b, s["cx"], s["cy"], 0, 0.30 * s["r"])
    face = _ring_mean(b, s["cx"], s["cy"], 1.20 * s["r"], 1.55 * s["r"])
    return None not in (cav, hub, face) and hub - cav >= 20 and face - cav >= 35


def find_parts(bgr, min_frac=0.025, max_frac=0.30, dark=0.25, want_mask=False):
    """Every impeller in a frame, as dicts with the crop square to cut.

    min/max_frac bound the cavity radius as a fraction of the frame's short
    side. 'dark' sets the threshold between the darkest pixels and the
    typical (background) ones; lower is stricter."""
    g0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    short = min(g0.shape[:2])
    scale = min(1.0, 600.0 / short)           # find on a smaller copy: faster
    g = cv2.resize(g0, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1 else g0
    s_short = min(g.shape[:2])
    min_r, max_r = min_frac * s_short, max_frac * s_short

    b = cv2.GaussianBlur(g, (0, 0), max(1.0, min_r / 30))
    # 'darkest' = the level of the darkest patch the size of half the
    # smallest cavity allowed. A plain percentile would drift with how much
    # of the frame is background, and on a sparse frame would land on the
    # shaded half of a part's face instead of inside the cavity.
    k = int(min(b.size - 1, max(50, 0.5 * np.pi * min_r ** 2)))
    lo = float(np.partition(b.ravel(), k)[k])
    hi = float(np.percentile(b, 60))
    thr = lo + dark * (hi - lo)
    mask = (b < thr).astype(np.uint8) * 255
    k = max(3, int(min_r * 0.12)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    found = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < np.pi * min_r ** 2 or area > np.pi * (1.4 * max_r) ** 2:
            continue
        # A shaded side of the face or the rim's shadow can join the cavity
        # as a thin dark arc. Opening with a disc removes arcs but keeps a
        # round cavity, as long as the disc is narrower than the cavity; try
        # gentle first, then harder, and keep the first result that passes.
        x, y, w, h = cv2.boundingRect(c)
        filled = np.zeros((h + 4, w + 4), np.uint8)
        cv2.drawContours(filled, [c - [x - 2, y - 2]], -1, 255, -1)
        r_raw = np.sqrt(area / np.pi)
        for frac in (0.5, 0.7, 0.9):
            kk = max(5, int(frac * r_raw)) | 1
            sub = cv2.morphologyEx(filled, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
            inner, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cands = [_shape(c2, (x - 2, y - 2)) for c2 in inner
                     if cv2.contourArea(c2) >= np.pi * min_r ** 2]
            cands = [q for q in cands if _round_enough(q) and min_r <= q["r"] <= max_r
                     and _looks_like_impeller(b, q)]
            if cands:
                found.append(max(cands, key=lambda q: q["r"]))
                break

    # one part per cavity: drop any that sit inside a bigger one
    found.sort(key=lambda q: -q["r"])
    kept = []
    for q in found:
        if all(np.hypot(q["cx"] - p["cx"], q["cy"] - p["cy"]) > 0.8 * p["r"] for p in kept):
            kept.append(q)

    H, W = g0.shape[:2]
    parts = []
    for q in kept:
        cx, cy, r = q["cx"] / scale, q["cy"] / scale, q["r"] / scale
        side = int(round(CROP_PER_RADIUS * r))
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        inside = (max(0, min(W, x0 + side) - max(0, x0)) *
                  max(0, min(H, y0 + side) - max(0, y0))) / float(side * side)
        parts.append({"cx": cx, "cy": cy, "r": r, "box": (x0, y0, side),
                      "in_frame": inside})
    parts = reading_order(parts)
    if want_mask:
        return parts, cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return parts


RIM_PER_CAVITY = 1.80     # measured on the dataset; see make_detector_data.py


class CavityFinder:
    """The original finder: dark round cavities. No training, but shadows and
    grey prints can hide a part from it."""

    def __init__(self, args):
        self.args = args
        self.name = "dark-cavity finder"

    def __call__(self, frame, want_mask=False):
        a = self.args
        return find_parts(frame, a.min_size, a.max_size, a.dark, want_mask)


class YoloFinder:
    """The trained part detector (train_detector.py), on the GPU. It draws a
    box round each part's rim; the crop the classifiers see is then sized
    from the rim the same way the dataset photos were framed."""

    def __init__(self, weights, conf=0.5, imgsz=832, crop_per_rim_diameter=1.236):
        from ultralytics import YOLO
        import torch
        self.model = YOLO(str(weights))
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.conf, self.imgsz, self.k = conf, imgsz, crop_per_rim_diameter
        self.name = f"trained detector ({weights})"

    def __call__(self, frame, want_mask=False):
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                 device=self.device, verbose=False)[0]
        xyxy = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
        conf = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros(0)
        parts = boxes_to_parts(frame.shape, xyxy, conf, self.k)
        return (parts, None) if want_mask else parts


def boxes_to_parts(shape, xyxy, conf, k=1.236):
    """Rim boxes -> the same part records find_parts() returns."""
    H, W = shape[:2]
    parts = []
    for (x0, y0, x1, y1), c in zip(xyxy, conf):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        diam = ((x1 - x0) + (y1 - y0)) / 2
        side = int(round(k * diam))
        bx, by = int(round(cx - side / 2)), int(round(cy - side / 2))
        inside = (max(0, min(W, bx + side) - max(0, bx)) *
                  max(0, min(H, by + side) - max(0, by))) / float(side * side)
        # a box against the frame edge is a part cut off by it: its size and
        # centre are wrong, so it is shown but never classified as whole
        if x0 <= 2 or y0 <= 2 or x1 >= W - 3 or y1 >= H - 3:
            inside = min(inside, 0.5)
        parts.append({"cx": float(cx), "cy": float(cy),
                      "r": float(diam / 2 / RIM_PER_CAVITY),
                      "box": (bx, by, side), "in_frame": inside,
                      "conf": float(c)})
    return reading_order(parts)


def make_finder(args):
    """The trained detector when it exists, otherwise the cavity finder."""
    w = Path(args.detector)
    if args.finder == "cavity" or (args.finder == "auto" and not w.is_file()):
        return CavityFinder(args)
    if not w.is_file():
        raise SystemExit(f"Missing {w}\nTrain it:  python train_detector.py")
    try:
        meta = json.loads((w.parent / "detector.json").read_text())
    except (OSError, ValueError):
        meta = {}
    try:
        return YoloFinder(w, args.det_conf, meta.get("imgsz", 832),
                          meta.get("crop_per_rim_diameter", 1.236))
    except ImportError:
        if args.finder == "yolo":
            raise SystemExit("Ultralytics is not installed:  pip install ultralytics")
        print("Ultralytics not installed -- using the dark-cavity finder.")
        return CavityFinder(args)


def reading_order(items):
    """Rows top to bottom, then left to right. A new row starts when a part
    sits more than half a part-height below the first part of the current
    row -- not at fixed bands, which would flip two parts' order whenever one
    of them wobbled across a band edge."""
    if not items:
        return items
    size = np.median([q["r"] for q in items]) * CROP_PER_RADIUS
    rows, cur = [], []
    for q in sorted(items, key=lambda q: q.get("sy", q["cy"])):
        if cur and q.get("sy", q["cy"]) - cur[0].get("sy", cur[0]["cy"]) > 0.5 * size:
            rows.append(cur)
            cur = []
        cur.append(q)
    rows.append(cur)
    return [q for row in rows for q in sorted(row, key=lambda q: q.get("sx", q["cx"]))]


def cut(bgr, box):
    """The crop square; anything past the frame edge is filled by repeating
    the edge pixels so no hard black border appears."""
    x0, y0, side = box
    H, W = bgr.shape[:2]
    pad = max(0, -x0, -y0, x0 + side - W, y0 + side - H)
    if pad:
        bgr = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0 = x0 + pad, y0 + pad
    return bgr[y0:y0 + side, x0:x0 + side]


# --------------------------------------------------------------------------
# following parts from frame to frame (live only)
# --------------------------------------------------------------------------

class Tracker:
    """Gives each part a number that stays put while it stays put, and
    averages its scores over frames so the verdict does not flicker."""

    def __init__(self, alpha=0.35, forget=30):
        self.alpha, self.forget = alpha, forget
        self.tracks, self.next_id = [], 1

    def update(self, parts, p1, p2):
        used = set()
        for t in self.tracks:
            t["missed"] += 1
        for i, p in enumerate(parts):
            best, bd = None, None
            for t in self.tracks:
                if id(t) in used:
                    continue
                d = np.hypot(p["cx"] - t["cx"], p["cy"] - t["cy"])
                # two parts' centres are at least ~3.4 cavity radii apart
                # (their rims would touch), so anything within 1.5 radii of
                # where a part was is that same part
                if d < 1.5 * max(p["r"], t["r"]) and (bd is None or d < bd):
                    best, bd = t, d
            pd = float(p1[i])
            pt = None if p2 is None else np.asarray(p2[i], float)
            if best is None:
                best = {"id": self.next_id, "age": 0, "logged": False,
                        "p_def": pd, "p_types": pt}
                self.next_id += 1
                self.tracks.append(best)
            else:
                a = self.alpha
                best["p_def"] = a * pd + (1 - a) * best["p_def"]
                if pt is not None:
                    best["p_types"] = (pt if best["p_types"] is None
                                       else a * pt + (1 - a) * best["p_types"])
            best.update(cx=p["cx"], cy=p["cy"], r=p["r"], box=p["box"],
                        in_frame=p["in_frame"], missed=0, conf=p.get("conf"))
            # a slow-moving position, used only to decide the numbering
            best["sx"] = 0.8 * best.get("sx", p["cx"]) + 0.2 * p["cx"]
            best["sy"] = 0.8 * best.get("sy", p["cy"]) + 0.2 * p["cy"]
            best["age"] += 1
            used.add(id(best))
        self.tracks = [t for t in self.tracks if t["missed"] <= self.forget]
        # Two tracks on one part (it jumped further than usual while the old
        # track was waiting for it): keep the one seen longest, carry over
        # whether it was already recorded, drop the other.
        self.tracks.sort(key=lambda t: -t["age"])
        merged = []
        for t in self.tracks:
            twin = next((m for m in merged
                         if np.hypot(t["cx"] - m["cx"], t["cy"] - m["cy"])
                         < 1.5 * max(t["r"], m["r"])), None)
            if twin is None:
                merged.append(t)
                continue
            twin["logged"] = twin["logged"] or t["logged"]
            if t["missed"] < twin["missed"]:          # newer sighting wins
                for k in ("cx", "cy", "r", "box", "in_frame", "missed",
                          "p_def", "p_types"):
                    twin[k] = t[k]
        self.tracks = merged
        # Numbered 1, 2, 3 ... in reading order for display and the record:
        # the internal id keeps climbing as parts come and go, which is
        # useless for saying "part 3 is wrong". Parts the finder missed for a
        # frame or two keep their place, so the numbers do not shuffle.
        self.tracks = reading_order(self.tracks)
        for n, t in enumerate(self.tracks, start=1):
            t["pos"] = n
        return [t for t in self.tracks if t["missed"] == 0]


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------

class Session:
    """scans/<stamp>/ with parts.csv, session.json and crops/."""

    def __init__(self, root, source_meta, cuts):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / stamp
        (self.dir / "crops").mkdir(parents=True, exist_ok=True)
        self.name, self.meta, self.cuts = stamp, source_meta, cuts
        self.records, self.scans = [], 0
        self.csv = self.dir / "parts.csv"
        with open(self.csv, "w", newline="") as f:
            csv.writer(f).writerow(
                ["scan", "part", "time", "source", "verdict", "confidence",
                 "p_defective"] + [f"p_{fl}" for fl in FLAGS] +
                ["defects", "label", "correct", "crop", "x", "y", "size"])

    def new_scan(self):
        self.scans += 1
        return self.scans

    def add(self, scan_no, part_no, crop_bgr, result, box, source, label=None):
        ident = f"scan{scan_no:04d}-p{part_no}"
        crop_name = f"crops/{ident}.png"
        cv2.imwrite(str(self.dir / crop_name), crop_bgr)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        verdict = "defective" if result["defective"] else "good"
        types = result["types"] if result["defective"] else None
        defects = None if types is None else [
            {"type": fl, "score": round(types[fl], 4),
             "present": types[fl] >= self.cuts[fl],
             "cutoff": round(self.cuts[fl], 2)} for fl in FLAGS]
        rec = {
            "id": ident, "file": crop_name, "split": None,
            "label": label, "verdict": verdict,
            "confidence": round(result["confidence"], 4),
            "correct": None if label is None else (label == verdict),
            "defects": defects,
            "defect_count": None if defects is None else
            sum(1 for d in defects if d["present"]),
            "hand_labelled": False, "duplicate": None,
            "scan": {"session": self.name, "scan": scan_no, "part": part_no,
                     "time": now, "source": source,
                     "box": [int(v) for v in box]},
        }
        self.records.append(rec)
        all_types = result["types"] or {}
        with open(self.csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [scan_no, part_no, now, source, verdict, f"{result['confidence']:.4f}",
                 f"{result['p_def']:.4f}"] +
                [f"{all_types[fl]:.4f}" if fl in all_types else "" for fl in FLAGS] +
                [";".join(d["type"] for d in defects or [] if d["present"]),
                 label or "", "" if rec["correct"] is None else rec["correct"],
                 crop_name, int(box[0]), int(box[1]), int(box[2])])
        return rec

    def write_json(self):
        try:
            from export_results import summarise, SCHEMA
        except ImportError:          # export_results.py missing: same shape, local
            summarise, SCHEMA = None, "casting-inspection/1"
        doc = {
            "schema": SCHEMA,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": dict(self.meta, session=self.name, scans=self.scans),
            "summary": (summarise(self.records, self.cuts) if summarise
                        else {"castings": len(self.records)}),
            "castings": self.records,
        }
        tmp = self.dir / "session.json.tmp"
        tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        os.replace(tmp, self.dir / "session.json")   # never half-written

    def tally(self):
        good = sum(1 for r in self.records if r["verdict"] == "good")
        bad = len(self.records) - good
        types = {fl: sum(1 for r in self.records for d in (r["defects"] or [])
                         if d["type"] == fl and d["present"]) for fl in FLAGS}
        return good, bad, types


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def draw(frame, tracks, session, info, doubtful, mask=None, cuts=None, sel_id=None):
    cuts = cuts or {fl: 0.5 for fl in FLAGS}
    scale = VIEW_HEIGHT / frame.shape[0]
    view = cv2.resize(frame, (max(1, int(frame.shape[1] * scale)), VIEW_HEIGHT))
    if mask is not None:
        small = cv2.resize(mask, (view.shape[1] // 3, VIEW_HEIGHT // 3),
                           interpolation=cv2.INTER_NEAREST)
        view[:small.shape[0], -small.shape[1]:] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    for t in tracks:
        x0, y0, side = (int(v * scale) for v in t["box"])
        res = t["result"]
        settled = t["age"] >= 3
        col = L.GREY if not settled else (L.RED if res["defective"] else L.GREEN)
        cv2.rectangle(view, (x0, y0), (x0 + side, y0 + side), col, 2)
        if t["id"] == sel_id:          # the part being looked at in the panel
            cv2.rectangle(view, (x0 - 5, y0 - 5), (x0 + side + 5, y0 + side + 5),
                          YELLOW, 3)
        word = "DEFECTIVE" if res["defective"] else "GOOD"
        head = f"#{t['pos']} {word} {100*res['confidence']:.0f}%"
        found = ", ".join(fl + ("?" if fl in doubtful else "") for fl in res["found"])
        if found:
            head += "  " + found
        if mask is None and info.get("show_conf") and t.get("conf") is not None:
            head += f"  [found {t['conf']:.2f}]"
        if t["in_frame"] < 0.9:
            head += "  (move into view)"
        elif t.get("logged"):
            head += "  - saved"
        ty = y0 - 10 if y0 > 30 else y0 + side + 24
        L._text(view, head, (x0 + 2, ty), 0.6, col, 2)

    panel = np.full((VIEW_HEIGHT, PANEL_WIDTH, 3), L.DARK, np.uint8)
    y = 46
    L._text(panel, f"{len(tracks)} part{'s' if len(tracks) != 1 else ''} in view",
            (20, y), 0.9, L.WHITE, 2)
    y += 30
    if not tracks:
        for line in ("none found -- light plain background,",
                     "parts not touching, cavity visible"):
            L._text(panel, line, (20, y), 0.48, L.GREY)
            y += 20
    # One block per part: its verdict, then a bar for each defect type, the
    # same bars as live_inspect.py -- so a part with two defects shows two
    # bars over their cutoffs (the white ticks), not just a list of names.
    # Eight at a time; the window follows the selected part, so with 15 in
    # view the arrow keys or the mouse wheel walk through all of them.
    PER = 8
    ids = [t["id"] for t in tracks]
    sel = ids.index(sel_id) if sel_id in ids else 0
    first = max(0, min(sel - PER // 2, len(tracks) - PER))
    shown = tracks[first:first + PER]
    if len(tracks) > PER:
        L._text(panel, f"showing {first + 1}-{first + len(shown)} of {len(tracks)}"
                "   up/down or wheel", (20, y - 6), 0.42, L.GREY)
        y += 14
    block = min(58, (500 - y) // max(1, len(shown)))
    bar_w, gap, x0 = 78, 12, 20
    for t in shown:
        res = t["result"]
        col = L.RED if res["defective"] else L.GREEN
        if t["id"] == sel_id and len(tracks) > 1:
            cv2.rectangle(panel, (4, y - 18), (PANEL_WIDTH - 4, y + block - 22),
                          YELLOW, 2)
        L._text(panel, f"#{t['pos']}", (x0, y), 0.55, L.WHITE, 1)
        L._text(panel, ("DEFECTIVE " if res["defective"] else "GOOD ") +
                f"{100*res['confidence']:.0f}%", (x0 + 46, y), 0.55, col, 2)
        if res["defective"] and res["types"]:
            for k, fl in enumerate(FLAGS):
                bx = x0 + k * (bar_w + gap)
                p, hit, doubt = res["types"][fl], fl in res["found"], fl in doubtful
                c = (L.AMBER if doubt else L.RED) if hit else L.GREY
                L._text(panel, f"{fl}{'?' if doubt else ''} {p:.2f}", (bx, y + 18),
                        0.38, c, 1)
                cv2.rectangle(panel, (bx, y + 24), (bx + bar_w, y + 32), (70, 70, 70), -1)
                cv2.rectangle(panel, (bx, y + 24), (bx + int(bar_w * p), y + 32), c, -1)
                tick = bx + int(bar_w * cuts[fl])
                cv2.line(panel, (tick, y + 21), (tick, y + 35), L.WHITE, 1)
        elif res["defective"]:
            L._text(panel, "defect-type model not loaded", (x0, y + 20), 0.42, L.GREY)
        else:
            L._text(panel, "no defect-type check needed", (x0, y + 20), 0.42, L.GREY)
        y += block
    if shown and any(t["result"]["defective"] for t in shown):
        L._text(panel, "bar past the white tick = that defect is called", (x0, y),
                0.4, L.GREY)
        y += 16

    y = max(y + 14, 520)
    good, bad, types = session.tally()
    L._text(panel, f"Recorded: {good + bad}", (20, y), 0.6, L.WHITE, 2)
    L._text(panel, f"good {good}", (170, y), 0.55, L.GREEN, 2)
    L._text(panel, f"defective {bad}", (265, y), 0.55, L.RED, 2)
    y += 24
    L._text(panel, "   ".join(f"{fl}{'?' if fl in doubtful else ''} {types[fl]}"
                              for fl in FLAGS), (20, y), 0.5, L.WHITE)
    y += 20
    L._text(panel, f"{session.scans} scans -> {session.dir}", (20, y), 0.4, L.GREY)

    yb = VIEW_HEIGHT - 60
    L._text(panel, info.get("source", ""), (20, yb), 0.45, L.GREY)
    status = [f"{info['fps']:.0f} fps"] if info.get("fps") else []
    status.append("AUTO-LOG" if info.get("auto") else "press C to record")
    if info.get("paused"):
        status.append("PAUSED")
    L._text(panel, "   ".join(status), (20, yb + 20), 0.45, L.GREY)
    L._text(panel, "c scan  f frame  space pause  d finder  click/up/down pick  q quit",
            (20, yb + 40), 0.42, L.GREY)
    return np.hstack([view, panel])


# --------------------------------------------------------------------------
# classifying a batch of found parts
# --------------------------------------------------------------------------

def classify(inspector, frame, parts):
    """-> crops, p_defective array, defect-score array (or None)"""
    crops = [cut(frame, p["box"]) for p in parts]
    if not crops:
        return [], np.zeros(0), None
    p1, p2 = inspector.predict([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops])
    return crops, p1, p2


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------

def run_live(inspector, args, cuts, doubtful, session):
    print(f"Opening camera {args.camera} ...")
    cap, used, dark = L.open_camera(args.camera, args.backend)
    if cap is None:
        raise SystemExit(f"Camera {args.camera} did not open. "
                         "Try:  python live_inspect.py --list-cameras")
    print(f"Camera {args.camera} via {used}" + ("  -- BLACK" if dark else ""))
    win = "Multi-part inspection"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    events = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            events.append(("click", x, y))
        elif event == cv2.EVENT_MOUSEWHEEL:
            events.append(("step", -1 if cv2.getMouseWheelDelta(flags) > 0 else 1, 0))
    cv2.setMouseCallback(win, on_mouse)
    sel_id = None
    tracker = Tracker(args.smooth)
    paused, show_mask, fps, t_prev = False, False, 0.0, time.time()
    frame, live, crops_by_id, mask = None, [], {}, None

    while True:
        if not paused:
            ok, f = cap.read()
            if not ok:
                print("Camera stopped sending frames.")
                break
            frame = f
            out = args.finder_fn(frame, want_mask=show_mask)
            parts, mask = (out if show_mask else (out, None))
            crops, p1, p2 = classify(inspector, frame, parts)
            live = tracker.update(parts, p1, p2)
            crops_by_id = {}
            for t in live:
                t["result"] = L.decide(t["p_def"], t["p_types"], cuts, args.threshold)
                # crop of this frame for this track (same order as parts)
                for p, c in zip(parts, crops):
                    if p["box"] == t["box"]:
                        crops_by_id[t["id"]] = c
            if args.auto:
                fresh = [t for t in live if not t["logged"] and t["age"] >= args.settle
                         and t["in_frame"] >= 0.9]
                if fresh:
                    record(session, fresh, crops_by_id, f"camera {args.camera}")

        now = time.time()
        if not paused and now > t_prev:
            inst = 1.0 / (now - t_prev)
            fps = 0.9 * fps + 0.1 * inst if fps else inst
        t_prev = now
        ids = [t["id"] for t in live]
        if sel_id not in ids:
            sel_id = ids[0] if ids else None
        for kind, a, b in events:
            if kind == "step" and ids:
                sel_id = ids[(ids.index(sel_id) + a) % len(ids)]
            elif kind == "click" and frame is not None:
                sc = VIEW_HEIGHT / frame.shape[0]
                fx, fy = a / sc, b / sc
                for t in live:
                    bx, by, bs = t["box"]
                    if bx <= fx <= bx + bs and by <= fy <= by + bs:
                        sel_id = t["id"]
        events.clear()
        cv2.imshow(win, draw(frame, live, session,
                             {"source": f"camera {args.camera}", "fps": fps,
                              "paused": paused, "auto": args.auto,
                              "show_conf": show_mask},
                             doubtful, mask if show_mask else None, cuts, sel_id))
        key = cv2.waitKeyEx(1)
        k = key & 0xFF if key != -1 else 255
        if key in UP_KEYS:
            events.append(("step", -1, 0))
            continue
        if key in DOWN_KEYS:
            events.append(("step", 1, 0))
            continue
        if k in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if k == ord(" "):
            paused = not paused
        elif k == ord("d"):
            show_mask = not show_mask
        elif k == ord("f") and frame is not None:
            # the whole camera frame, untouched -- for working out why the
            # finder misses a part
            n = len(list(session.dir.glob("frame-*.png"))) + 1
            path = session.dir / f"frame-{n}.png"
            cv2.imwrite(str(path), frame)
            print(f"saved {path}")
        elif k == ord("c"):
            ready = [t for t in live if t["age"] >= 3 and t["in_frame"] >= 0.9]
            if ready:
                record(session, ready, crops_by_id, f"camera {args.camera}")
            else:
                print("Nothing to record: no settled part fully in view.")
    cap.release()
    cv2.destroyAllWindows()
    return session


def record(session, tracks, crops_by_id, source):
    scan = session.new_scan()
    for t in sorted(tracks, key=lambda t: t["pos"]):
        crop = crops_by_id.get(t["id"])
        if crop is None:
            continue
        session.add(scan, t["pos"], crop, t["result"], t["box"], source)
        t["logged"] = True
    session.write_json()
    good, bad, _ = session.tally()
    print(f"scan {scan}: {len(tracks)} part(s) recorded -- session total "
          f"{good + bad} ({good} good, {bad} defective)")


def run_folder(inspector, args, cuts, session):
    files = L.image_files(args.folder)
    if not files:
        raise SystemExit(f"No images under {args.folder}")
    none_found = []
    for path in files:
        bgr, _ = L.read_rgb(path)
        if bgr is None:
            continue
        if args.one_per_image:
            h, w = bgr.shape[:2]
            s = min(h, w)
            parts = [{"box": ((w - s) // 2, (h - s) // 2, s), "in_frame": 1.0}]
        else:
            parts = args.finder_fn(bgr)
        if not parts:
            none_found.append(path)
            continue
        crops, p1, p2 = classify(inspector, bgr, parts)
        scan = session.new_scan()
        label = L.true_label(path)
        for n, (p, c) in enumerate(zip(parts, crops), start=1):
            res = L.decide(float(p1[n - 1]), None if p2 is None else p2[n - 1],
                           cuts, args.threshold)
            # a folder's label applies to its images only when each image
            # holds one part, as in the dataset
            session.add(scan, n, c, res, p["box"], str(path),
                        label if len(parts) == 1 else None)
    session.write_json()
    good, bad, types = session.tally()
    print(f"\n{len(files)} images, {good + bad} parts: {good} good, {bad} defective")
    print("  defect types: " + ", ".join(f"{fl} {types[fl]}" for fl in FLAGS))
    judged = [r for r in session.records if r["correct"] is not None]
    if judged:
        right = sum(r["correct"] for r in judged)
        print(f"  agrees with the folder labels on {right}/{len(judged)} "
              f"({100*right/len(judged):.1f}%)")
    if none_found:
        print(f"  no part found in {len(none_found)} image(s), e.g. {none_found[0].name}"
              "\n  (dataset-style single photos: add --one-per-image)")
    print(f"Written to {session.dir}")
    return session


# --------------------------------------------------------------------------
# --selftest: frames with known answers
# --------------------------------------------------------------------------

def make_frame(images, rng, size=(720, 1280)):
    """Lay 2-6 dataset photos on a light background like parts on paper,
    each its own size, with mild camera blur and noise. Returns the frame and
    (which image, where it went) for each one that fitted."""
    H, W = size
    frame = np.full((H, W, 3), int(rng.integers(200, 235)), np.uint8)
    placed = []
    for idx, img in enumerate(images):
        for _ in range(80):
            side = int(rng.integers(int(0.30 * H), int(0.48 * H)))
            x0, y0 = int(rng.integers(0, W - side)), int(rng.integers(0, H - side))
            if all(x0 + side + 8 < bx or bx + bs + 8 < x0 or
                   y0 + side + 8 < by or by + bs + 8 < y0
                   for _, (bx, by, bs) in placed):
                break
        else:
            continue                                  # no room: leave it out
        frame[y0:y0 + side, x0:x0 + side] = cv2.resize(
            img, (side, side), interpolation=cv2.INTER_CUBIC)
        placed.append((idx, (x0, y0, side)))
    frame = cv2.GaussianBlur(frame, (0, 0), float(rng.uniform(0.4, 1.0)))
    noise = rng.normal(0, 3, frame.shape)
    frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return frame, placed


def run_selftest(inspector, args, cuts, frames=40, seed=0):
    files = L.image_files(args.selftest)
    labelled = [f for f in files if L.true_label(f)]
    if len(labelled) < 6:
        raise SystemExit(f"Need images under {args.selftest}\\def_front and "
                         f"\\ok_front, found {len(labelled)}")
    rng = np.random.default_rng(seed)
    out_dir = Path(args.out) / ("selftest-" + time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    placed_n = found_n = extra = 0
    same_as_single = right_multi = right_single = scored = 0
    for k in range(frames):
        pick = [labelled[i] for i in rng.choice(len(labelled),
                                                 int(rng.integers(2, 7)), replace=False)]
        imgs = [L.read_rgb(p)[0] for p in pick]
        frame, placed = make_frame(imgs, rng)
        parts = args.finder_fn(frame)
        placed_n += len(placed)
        # pair each placed photo with the found part whose centre lies in it
        matched, taken = [], set()
        for idx, (bx, by, bs) in placed:
            for i, p in enumerate(parts):
                if i not in taken and bx <= p["cx"] <= bx + bs and by <= p["cy"] <= by + bs:
                    matched.append((idx, i))
                    taken.add(i)
                    break
        found_n += len(matched)
        extra += len(parts) - len(matched)
        if k < 4:
            vis = frame.copy()
            for p in parts:
                x0, y0, s = p["box"]
                cv2.rectangle(vis, (x0, y0), (x0 + s, y0 + s), (0, 0, 255), 3)
            cv2.imwrite(str(out_dir / f"frame{k+1}.jpg"), vis)
        if not matched:
            continue
        crops, p1, _ = classify(inspector, frame, [parts[i] for _, i in matched])
        s1, _ = inspector.predict([cv2.cvtColor(imgs[idx], cv2.COLOR_BGR2RGB)
                                   for idx, _ in matched])
        for n, (idx, _) in enumerate(matched):
            truth = L.true_label(pick[idx]) == "defective"
            m_def, s_def = p1[n] >= args.threshold, s1[n] >= args.threshold
            scored += 1
            right_multi += m_def == truth
            right_single += s_def == truth
            same_as_single += m_def == s_def

    print()
    print("=" * 66)
    print(f"  SELF-TEST: {frames} frames, {placed_n} parts placed")
    print("=" * 66)
    print(f"  found          {found_n}/{placed_n}  ({100*found_n/max(1, placed_n):.1f}%)")
    print(f"  false finds    {extra}")
    if scored:
        print(f"  verdict right  {right_multi}/{scored} found in a group "
              f"({100*right_multi/scored:.1f}%)")
        print(f"                 {right_single}/{scored} same photos one at a time "
              f"({100*right_single/scored:.1f}%)")
        print(f"  same verdict   {same_as_single}/{scored} as one at a time")
    print(f"  pictures of the first frames with what was found: {out_dir}")
    ok = placed_n and found_n / placed_n >= 0.95 and extra == 0 and \
        (not scored or same_as_single / scored >= 0.97)
    print()
    print("  PASS -- finding and cutting out parts does not change the answers"
          if ok else
          "  CHECK -- see the numbers above and the saved frames")
    print("=" * 66)
    return 0 if ok else 1


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int)
    src.add_argument("--folder", metavar="DIR", help="every image in a folder")
    src.add_argument("--selftest", metavar="DIR",
                     help="a folder with def_front/ and ok_front/ inside")
    ap.add_argument("--auto", action="store_true",
                    help="live: record each part once, after it has been "
                         "steady for --settle frames")
    ap.add_argument("--settle", type=int, default=15)
    ap.add_argument("--one-per-image", action="store_true",
                    help="--folder: treat each image as one part, as in the "
                         "dataset, instead of searching it")
    ap.add_argument("--min-size", type=float, default=0.025,
                    help="smallest cavity radius, as a fraction of the frame's "
                         "short side")
    ap.add_argument("--max-size", type=float, default=0.30)
    ap.add_argument("--dark", type=float, default=0.25,
                    help="how dark counts as the cavity, 0-1 (lower = stricter)")
    ap.add_argument("--finder", choices=["auto", "yolo", "cavity"], default="auto",
                    help="how parts are found: 'auto' uses the trained detector "
                         "when detector\\impeller.pt exists")
    ap.add_argument("--detector", default="detector/impeller.pt")
    ap.add_argument("--det-conf", type=float, default=0.5,
                    help="detector: how sure it must be to call something a part")
    ap.add_argument("--backend", choices=["auto", "dshow", "msmf", "any"],
                    default="auto")
    ap.add_argument("--cut", choices=["f1", "rate", "half"], default="f1")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--smooth", type=float, default=0.35)
    ap.add_argument("--out", default="scans", help="where session folders go")
    args = ap.parse_args()

    run_dir = L.newest_stage_one()
    if run_dir is None:
        raise SystemExit("No trained stage one found under results\\.")
    s1 = run_dir / "model_parameters.pth"
    names, acc, n = L.stage_one_info(run_dir)
    s2, meta, thr = L.find_stage_two()
    cuts, cut_src = L.load_cuts(thr, args.cut)
    doubtful = {}
    print(f"Stage one : {s1}" + (f"  ({acc}% held out)" if acc is not None else ""))
    print(f"Stage two : {s2 or 'not found -- run apply_defect_types.py --no-pos-weight'}")
    print("Loading ...")
    inspector = L.Inspector(s1, names, s2, meta)
    print(f"Running on {inspector.device}")
    args.finder_fn = make_finder(args)
    print(f"Finding parts with the {args.finder_fn.name}")

    if args.selftest:
        return run_selftest(inspector, args, cuts)

    source_meta = {"stage_one": str(s1), "stage_one_accuracy_pct": acc,
                   "stage_two": str(s2) if s2 else None, "cutoffs": cut_src,
                   "mode": "camera" if args.camera is not None else "folder",
                   "finder": args.finder_fn.name}
    session = Session(args.out, source_meta, cuts)
    print(f"Recording to {session.dir}")
    if args.folder:
        run_folder(inspector, args, cuts, session)
    else:
        run_live(inspector, args, cuts, doubtful, session)
        session.write_json()
        good, bad, _ = session.tally()
        print(f"Session saved: {good + bad} parts ({good} good, {bad} defective) "
              f"in {session.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
