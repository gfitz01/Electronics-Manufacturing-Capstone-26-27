"""Test what the discovered grouping tracks, using narrow defect flags.

    python label_check.py --build                  make the blind sheet
    python label_check.py --score my_labels.csv    score your answers

Round 1 and why this round exists
---------------------------------
The first pass used two flags: "frayed / torn edge" and "spots, chips or
nicks". The second caught 89% of defective castings. A flag that is true 9
times out of 10 cannot sort anything -- there is almost no variance in it to
correlate with, so the test had no chance of detecting a real association even
if one existed.

That round still produced a solid result. Neither individual flag survived
correction for running two tests (p = 0.029 and 0.026 against a 0.025
threshold), but defects-per-casting differed clearly: 1.56 in group A against
1.19 in group B, difference +0.37 [95% CI +0.15 to +0.59]. Both flags also
moved in the SAME direction, which is the signature of a severity axis rather
than a fault taxonomy -- a taxonomy would push them opposite ways.

This round splits the overbroad question into narrow ones with explicit
exclusions, to find out whether any single well-defined defect discriminates
where the broad one could not.

The primary test is pre-registered
----------------------------------
This round exists to test one thing: whether a tightly defined "spot on the
face" discriminates. That is the PRIMARY hypothesis, tested at 0.05.

The other flags are exploratory and reported against a corrected threshold.
Declaring this in advance is what stops it being four shots at the same target
-- run four tests and keep whichever looks best and you will find something
about 19% of the time from pure noise.

Do not change --primary after seeing the results. That is the whole point of
fixing it here.

Rounds are kept separate
------------------------
Files are tagged (--tag, default r2), so round one's key and answers stay
where they are. The two rounds use different definitions and are not
poolable; they are separate measurements.
"""

import argparse
import base64
import csv
import io
import json
import math
import random
import sys
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# key, column name, button text, and the definition shown on the sheet.
# Narrow definitions with explicit exclusions -- that is the whole change.
FLAGS = [
    ("fray", "1", "frayed edge",
     "The rim is ragged, feathered or torn ALONG ITS LENGTH — a "
     "continuous disturbed edge. Not a single bite out of it."),
    ("chip", "2", "chip in rim",
     "Material actually MISSING from the edge: a discrete bite, notch or "
     "broken-away piece with its own boundary. One localised place, not a "
     "run of damage."),
    ("spot", "3", "spot on face",
     "A discrete dark or light blemish ON THE FLAT FACE, away from the rim "
     "— porosity, an inclusion, a blotch. NOT a scratch, NOT anything "
     "on the edge itself."),
    ("slash", "4", "slash / scratch",
     "A linear mark: a scratch, gouge or drag line. Longer than it is wide. "
     "Your own word for it — the thing that is a slash rather than a "
     "bad spot."),
]
FLAG_NAMES = [f[0] for f in FLAGS]


def defect_images(base, class_name):
    base = Path(base)
    found = []
    for split in ("train", "val"):
        directory = base / split / class_name
        if directory.is_dir():
            found.extend(sorted(
                p for p in directory.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES))
    return found


def image_data_uri(path, size=380, quality=88):
    with Image.open(path) as im:
        im = im.convert("L")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def resolve_grouping(paths, labels, args):
    """Which cluster is which. Names are deliberately neutral this round:
    round one showed the descriptive names were backwards (the group called
    surface_spot had FEWER spots than the one called frayed_edge)."""
    assignments = Path(args.assignments)
    sizes = Counter(int(x) for x in labels)
    if assignments.is_file():
        previous = {}
        with open(assignments, newline="") as f:
            for row in csv.DictReader(f):
                previous[str(Path(row["file"]))] = int(row["cluster"])
        tally = Counter()
        for i, p in enumerate(paths):
            if previous.get(str(p)) == args.anchor_cluster:
                tally[int(labels[i])] += 1
        if tally:
            target = tally.most_common(1)[0][0]
            return {target: "B", 1 - target: "A"}, sizes
    target = 0 if sizes[0] > sizes[1] else 1
    return {target: "B", 1 - target: "A"}, sizes


def wilson(successes, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe(a, na, b, nb):
    if na == 0 or nb == 0:
        return (0.0, 0.0, 0.0)
    p1, p2 = a / na, b / nb
    l1, u1 = wilson(a, na)
    l2, u2 = wilson(b, nb)
    d = p1 - p2
    return (d,
            max(-1.0, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)),
            min(1.0, d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)))


def fisher_two_sided(a, b, c, d):
    """Exact test on a 2x2. Used instead of a normal approximation because
    several of these cells are small -- round one had a 48/50 cell."""
    n = a + b + c + d
    r1, r2, c1 = a + b, c + d, a + c
    if min(r1, r2, c1, n - c1) < 0 or n == 0:
        return 1.0
    def p(x):
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)
    obs = p(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1)
                        if p(x) <= obs * (1 + 1e-9)))


def welch(xs, ys):
    """Difference in means with a 95% interval. For defects-per-casting."""
    na, nb = len(xs), len(ys)
    if na < 2 or nb < 2:
        return (0.0, 0.0, 0.0)
    ma, mb = sum(xs) / na, sum(ys) / nb
    va = sum((x - ma) ** 2 for x in xs) / (na - 1)
    vb = sum((y - mb) ** 2 for y in ys) / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    d = ma - mb
    return (d, d - 1.96 * se, d + 1.96 * se)


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build(args):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    if args.primary not in FLAG_NAMES:
        raise SystemExit(f"--primary must be one of {FLAG_NAMES}")

    features_path = Path(args.features)
    if not features_path.is_file():
        raise SystemExit(f"Missing {features_path}\n"
                         "Run:  python discover_defect_types.py")

    paths = defect_images(args.data, args.class_name)
    features = np.load(features_path)
    if len(paths) != features.shape[0]:
        raise SystemExit("features.npy and the dataset disagree on image "
                         "count. Re-run discover_defect_types.py")

    n_components = min(50, features.shape[0], features.shape[1])
    reduced = PCA(n_components=n_components, random_state=0).fit_transform(features)
    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(reduced)

    grouping, sizes = resolve_grouping(paths, labels, args)
    print("Groups (neutral names this round -- the descriptive ones were "
          "backwards):")
    for label in (0, 1):
        print(f"  cluster {label}  ->  group {grouping[label]}   "
              f"{sizes[label]:>5} castings")
    print()

    rng = random.Random(args.seed)
    if args.from_shortlist:
        # Label what the model cannot call, not a random sample. The shortlist
        # comes from apply_defect_types.py, ranked by how close the prediction
        # sits to 0.5. These are worth 2-3x random castings because each one
        # resolves a case the model currently gets wrong half the time.
        sl = Path(args.from_shortlist)
        if not sl.is_file():
            raise SystemExit(f"Missing {sl}\nRun:  python apply_defect_types.py")
        want, seen = [], set()
        with open(sl, newline="") as f:
            for row in csv.DictReader(f):
                nm = Path(row["file"].replace("\\", "/")).name
                if nm not in seen:
                    seen.add(nm); want.append(nm)
        byname = {p.name: i for i, p in enumerate(paths)}
        picked = [(byname[nm], int(labels[byname[nm]]))
                  for nm in want[:args.n] if nm in byname]
        if not picked:
            raise SystemExit(f"None of {sl}'s castings are in {args.data}.")
        print(f"  From shortlist: {len(picked)} castings the model finds")
        print("  ambiguous, instead of a random sample.\n")
        rng.shuffle(picked)
        # A shortlist round is deliberately NOT balanced across the two
        # groups -- it takes whatever the model finds ambiguous. Record that
        # rather than a per-group count that would not be true.
        per_group = None
        sampling = "shortlist"
    else:
        per_group = args.n // 2
        picked = []
        for group in (0, 1):
            pool = list(np.flatnonzero(labels == group))
            if len(pool) < per_group:
                raise SystemExit(f"cluster {group} has only {len(pool)} images")
            picked.extend((i, group) for i in rng.sample(pool, per_group))
        rng.shuffle(picked)
        sampling = "balanced"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    with open(out / f"blind_key{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "file", "cluster", "group"])
        for n, (i, group) in enumerate(picked, start=1):
            w.writerow([n, str(paths[i]), group, grouping[group]])

    (out / f"blind_meta{tag}.json").write_text(json.dumps({
        "tag": args.tag,
        "flags": FLAG_NAMES,
        "primary": args.primary,
        "grouping": {str(k): v for k, v in grouping.items()},
        "group_sizes": {grouping[k]: int(v) for k, v in sizes.items()},
        "total_defects": int(len(paths)),
        "sampling": sampling,
        "sample_per_group": per_group,
        "sample_by_group": {grouping[g]: n for g, n
                            in Counter(g for _, g in picked).items()},
        "seed": args.seed,
    }, indent=2), encoding="utf-8")

    cards = ",\n".join(
        f'{{"id":{n},"src":"{image_data_uri(paths[i], args.size)}"}}'
        for n, (i, _) in enumerate(picked, start=1))

    toggles = "\n".join(
        f'''    <button class="tog" id="tog-{name}" onclick="toggle('{name}')">
      <kbd>{key}</kbd><span class="tname">{text}</span>
      <span class="tdef">{definition}</span>
    </button>''' for name, key, text, definition in FLAGS)

    keymap = "\n".join(
        f"  else if (e.key === '{key}') {{ toggle('{name}'); e.preventDefault(); }}"
        for name, key, _, _ in FLAGS)

    html = (BLIND_HTML
            .replace("__CARDS__", cards)
            .replace("__TOGGLES__", toggles)
            .replace("__KEYMAP__", keymap)
            .replace("__FLAGS__", json.dumps(FLAG_NAMES))
            .replace("__N__", str(len(picked)))
            .replace("__PRIMARY__", args.primary))
    (out / f"blind_labels{tag}.html").write_text(html, encoding="utf-8")

    print(f"  {out/f'blind_labels{tag}.html'}   <- open this")
    print(f"  {out/f'blind_key{tag}.csv'}       <- do NOT open until done")
    print()
    if per_group is not None:
        print(f"{len(picked)} castings, {per_group} from each group, shuffled "
              f"(seed {args.seed}).")
    else:
        by = Counter(grouping[g] for _, g in picked)
        print(f"{len(picked)} castings from the shortlist "
              f"({', '.join(f'{v} from {k}' for k, v in sorted(by.items()))}), "
              f"shuffled (seed {args.seed}).")
        print("This sample is NOT balanced across the groups by design -- it is")
        print("whatever the model found ambiguous. Use it for labels to train")
        print("on, not for the group comparison, which assumes balance.")
    print(f"Primary test, fixed now: {args.primary}. The rest are exploratory.")
    print()
    print("Keys: 1-4 toggle a defect on or off, SPACE or ENTER moves on,")
    print("      0 marks it unreadable. Toggling nothing means none of the")
    print("      four are present, which is a real answer.")
    print()
    print(f"Open it with:   start defect_types\\blind_labels{tag}.html")
    return 0


BLIND_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind defect labelling</title>
<style>
  :root { color-scheme: light dark;
    --bg:#F1F2F4; --card:#fff; --ink:#15181C; --muted:#6B747D;
    --rule:#D3D7DC; --accent:#1F5E8C; --ok:#2E7D4F; --on:#1F5E8C; }
  @media (prefers-color-scheme: dark) { :root {
    --bg:#131619; --card:#1B1F24; --ink:#E9EBEE; --muted:#8B949D;
    --rule:#2E343B; --accent:#6FAFD8; --ok:#5FBE88; --on:#2C6C9E; } }
  * { box-sizing:border-box }
  body { background:var(--bg); color:var(--ink); margin:0;
    padding:26px 20px 60px; font:15px/1.6 system-ui, sans-serif }
  .wrap { max-width:1000px; margin:0 auto }
  h1 { font-size:24px; margin:0 0 4px; letter-spacing:-.02em }
  .lede { color:var(--muted); margin:0 0 18px; max-width:68ch; font-size:14px }
  .bar { height:4px; background:var(--rule); border-radius:2px;
    margin:0 0 16px; overflow:hidden }
  .bar span { display:block; height:100%; background:var(--accent); width:0 }
  .card { background:var(--card); border:1px solid var(--rule);
    border-radius:4px; padding:20px }
  .split { display:grid; gap:22px; grid-template-columns:400px 1fr;
    align-items:start }
  @media (max-width:820px) { .split { grid-template-columns:1fr } }
  #shot { width:100%; border-radius:3px; border:1px solid var(--rule);
    background:#000; display:block }
  .meta { font:12px ui-monospace,monospace; color:var(--muted);
    margin:10px 0 0; text-align:center }
  .tog { display:block; width:100%; text-align:left; margin-bottom:8px;
    padding:9px 12px; border:1px solid var(--rule); background:var(--bg);
    color:var(--ink); border-radius:3px; cursor:pointer;
    font:14px system-ui,sans-serif }
  .tog:hover { border-color:var(--accent) }
  .tog.on { background:var(--on); border-color:var(--on); color:#fff }
  .tog kbd { font:11px ui-monospace,monospace; opacity:.7; margin-right:8px }
  .tname { font-weight:600 }
  .tdef { display:block; font-size:12px; line-height:1.45; opacity:.75;
    margin-top:3px }
  .tog.on .tdef { opacity:.85 }
  .go { display:flex; gap:9px; margin-top:14px; flex-wrap:wrap }
  .go button { flex:1; font:14px system-ui,sans-serif; padding:11px 14px;
    border:1px solid var(--rule); background:var(--bg); color:var(--ink);
    border-radius:3px; cursor:pointer }
  .go button.next { background:var(--accent); border-color:var(--accent);
    color:#fff; font-weight:600 }
  .hint { font-size:12px; line-height:1.5; color:var(--muted);
    margin:10px 2px 0; max-width:60ch }
  .back { margin-top:10px; font-size:13px; text-align:center }
  .back a { color:var(--accent); cursor:pointer }
  #done { display:none; text-align:center }
  textarea { width:100%; height:170px; font:12px ui-monospace,monospace;
    padding:10px; border:1px solid var(--rule); border-radius:3px;
    background:var(--bg); color:var(--ink); margin-top:12px }
  .how { background:var(--card); border-left:3px solid var(--accent);
    padding:14px 18px; border-radius:0 4px 4px 0; margin-bottom:18px;
    font-size:14px }
  .how p { margin:0 0 8px; max-width:72ch } .how p:last-child { margin:0 }
  code { font:12px ui-monospace,monospace; background:var(--bg);
    padding:1px 5px; border-radius:2px }
</style></head><body><div class="wrap">

<h1>Blind defect labelling</h1>
<p class="lede">__N__ defective castings in random order. Nothing here says
which group any of them came from.</p>

<div class="how">
  <p><strong>Tick every defect you can see, then press space.</strong> A
  casting can have several, or none of these four &mdash; ticking nothing is a
  real answer and means none of these four descriptions fit.</p>
  <p>The definitions are deliberately narrow, and the exclusions matter as
  much as the inclusions. Last round "spots, chips or nicks" caught 89% of
  castings, which is why it could not sort anything. If you find yourself
  ticking <em>spot on face</em> on almost everything again, the wording still
  needs tightening &mdash; tell me rather than pushing through.</p>
  <p>Go on first impression. Press <code>0</code> if the image is genuinely
  unreadable.</p>
</div>

<div class="bar"><span id="fill"></span></div>

<div class="card">
  <div class="split">
    <div>
      <img id="shot" alt="casting">
      <p class="meta" id="meta"></p>
    </div>
    <div>
__TOGGLES__
      <div class="go">
        <button class="next" onclick="advance()">space &nbsp;&rarr;</button>
        <button onclick="skip()"><kbd>0</kbd> image unusable</button>
      </div>
      <p class="hint">Nothing ticked + space = <strong>you can see it fine
      and none of the four are present</strong>. That is an answer, and it
      gets recorded. <code>0</code> is only for an image you genuinely
      cannot judge &mdash; blurred, cropped, too dark &mdash; and it is
      thrown away, not counted.</p>
      <p class="back"><a onclick="back()">&larr; back one</a></p>
    </div>
  </div>
</div>

<div class="card" id="done">
  <p style="color:var(--ok);font-size:17px;margin:0 0 4px"><strong>All
  __N__ labelled.</strong></p>
  <p class="meta">Save the file, then run<br>
  <code>python label_check.py --score my_labels.csv</code></p>
  <div class="go" style="max-width:420px;margin:16px auto 0">
    <button class="next" onclick="save()">save my_labels.csv</button>
    <button onclick="again()">start over</button>
  </div>
  <textarea id="csv" readonly
    title="fallback: copy this and save it as my_labels.csv"></textarea>
</div>

<script>
const CARDS = [__CARDS__];
const FLAGS = __FLAGS__;
const answers = {};
let at = 0;
let current = {};

function freshCurrent() {
  current = {};
  for (const f of FLAGS) current[f] = 0;
}

function paint() {
  for (const f of FLAGS) {
    const el = document.getElementById('tog-' + f);
    if (el) el.classList.toggle('on', !!current[f]);
  }
}

function render() {
  const card = document.querySelector('.card');
  if (at >= CARDS.length) {
    card.style.display = 'none';
    document.getElementById('done').style.display = 'block';
    document.getElementById('csv').value = csvText();
    document.getElementById('fill').style.width = '100%';
    return;
  }
  card.style.display = 'block';
  document.getElementById('done').style.display = 'none';
  document.getElementById('shot').src = CARDS[at].src;
  document.getElementById('meta').textContent =
    (at + 1) + ' of ' + CARDS.length;
  document.getElementById('fill').style.width =
    (100 * at / CARDS.length) + '%';
  const prev = answers[CARDS[at].id];
  if (prev && prev !== 'skip') { current = Object.assign({}, prev); }
  else { freshCurrent(); }
  paint();
}

function toggle(flag) {
  if (at >= CARDS.length) return;
  current[flag] = current[flag] ? 0 : 1;
  paint();
}

function advance() {
  if (at >= CARDS.length) return;
  answers[CARDS[at].id] = Object.assign({}, current);
  at++;
  render();
}

function skip() {
  if (at >= CARDS.length) return;
  answers[CARDS[at].id] = 'skip';
  at++;
  render();
}

function back() { if (at > 0) { at--; render(); } }

function again() {
  for (const k of Object.keys(answers)) delete answers[k];
  at = 0; render();
}

function csvText() {
  let out = 'id,' + FLAGS.join(',') + '\\n';
  for (const c of CARDS) {
    const a = answers[c.id];
    if (!a || a === 'skip') out += c.id + ',' + FLAGS.map(() => '?').join(',') + '\\n';
    else out += c.id + ',' + FLAGS.map(f => a[f] ? 1 : 0).join(',') + '\\n';
  }
  return out;
}

function save() {
  const blob = new Blob([csvText()], {type: 'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'my_labels.csv';
  a.click();
  URL.revokeObjectURL(a.href);
}

document.addEventListener('keydown', e => {
  if (e.key === ' ' || e.key === 'Enter') { advance(); e.preventDefault(); }
  else if (e.key === '0') { skip(); e.preventDefault(); }
__KEYMAP__
  else if (e.key === 'Backspace' || e.key === 'ArrowLeft') { back(); e.preventDefault(); }
});

freshCurrent();
render();
</script>
</div></body></html>"""


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------

def score(args):
    out = Path(args.out)
    tag = f"_{args.tag}" if args.tag else ""
    key_path = out / f"blind_key{tag}.csv"
    meta_path = out / f"blind_meta{tag}.json"
    if not key_path.is_file():
        raise SystemExit(f"Missing {key_path}. Run --build first"
                         f"{f' with --tag {args.tag}' if args.tag else ''}.")

    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    flags = meta.get("flags", FLAG_NAMES)
    primary = meta.get("primary", args.primary)
    group_sizes = meta.get("group_sizes") or {}
    total_defects = meta.get("total_defects", 0)

    key = {}
    with open(key_path, newline="") as f:
        for row in csv.DictReader(f):
            key[int(row["id"])] = row.get("group", str(row["cluster"]))

    mine = {}
    with open(args.score, newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if not set(flags) <= cols:
            raise SystemExit(
                f"That CSV has columns {sorted(cols)}, but this round expects "
                f"id,{','.join(flags)}.\nIt is probably from an earlier round. "
                f"Score it with the matching --tag, or re-run --build.")
        converted = 0
        for row in reader:
            vals = [row[f].strip() for f in flags]
            if all(v in ("0", "1") for v in vals):
                mine[int(row["id"])] = [int(v) for v in vals]
            elif args.zeros_are_clean and all(v == "?" for v in vals):
                # The labeller pressed 0 to mean "nothing wrong with this
                # one", not "cannot read it". Those are opposite meanings:
                # the first is data, the second is a missing value. Recorded
                # here as all-flags-absent, and counted so the reinterpretation
                # stays visible in the output rather than hiding in the totals.
                mine[int(row["id"])] = [0] * len(flags)
                converted += 1

    scored = [(key[i], mine[i]) for i in sorted(key) if i in mine]
    skipped = len(key) - len(scored)
    if not scored:
        raise SystemExit("No usable answers in that file.")

    groups = sorted({g for g, _ in scored})
    if len(groups) != 2:
        raise SystemExit("Need both groups represented.")
    ga, gb = groups
    n = len(scored)

    print("=" * 68)
    print(f"  round           {args.tag or '(untagged)'}"
          f"     primary test: {primary}")
    print(f"  labelled        {n} castings"
          f"{f'   ({skipped} skipped as unreadable)' if skipped else ''}")
    if converted:
        print(f"  reinterpreted   {converted} rows marked with 0 read as "
              "'no visible defect'")
    print("=" * 68)
    print()

    # ---------------- castings with nothing visibly wrong ----------------
    clean = sum(1 for _, m in scored if sum(m) == 0)
    if clean:
        lo, hi = wilson(clean, n)
        print(f"  NO VISIBLE DEFECT: {clean}/{n} = {100*clean/n:.1f}%  "
              f"[{100*lo:.0f} to {100*hi:.0f}]")
        print()
        print("    Every casting in this sample is labelled defective in the")
        print("    dataset, so this is castings where the ground-truth label")
        print("    says defective and a human inspector sees nothing wrong.")
        print("    Either the published labels carry noise, or the defect is")
        print("    real but not visible at this scale. Both matter: it caps")
        print("    what ANY classifier trained on these labels can be said to")
        print("    have learned, and it is worth stating next to the 100%")
        print("    binary accuracy rather than leaving the two unreconciled.")
        by_group = Counter(g for g, m in scored if sum(m) == 0)
        if len(by_group) == 2:
            print()
            print(f"    split by group: "
                  + "   ".join(f"{g} {by_group[g]}" for g in sorted(by_group)))
        print()

    # ---------------- base rates: did tightening work? ----------------
    print("  Base rates -- the thing this round was built to fix:")
    print()
    for k, flag in enumerate(flags):
        hits = sum(1 for _, m in scored if m[k] == 1)
        lo, hi = wilson(hits, n)
        note = ""
        if hits / n > 0.80:
            note = "   <- still too broad to discriminate"
        elif hits / n < 0.08:
            note = "   <- now too rare to test"
        elif 0.25 <= hits / n <= 0.75:
            note = "   <- good range"
        print(f"    {flag:<7} {100*hits/n:5.1f}%   ({hits}/{n})"
              f"   [{100*lo:.0f} to {100*hi:.0f}]{note}")
    print()
    print("    Round 1's broad 'spots, chips or nicks' was 88.8%. A flag only")
    print("    has power to discriminate when it is somewhere near the middle.")
    print()

    # ---------------- how many defects per casting ----------------
    counts = Counter(sum(m) for _, m in scored)
    print("  Defects per casting:")
    for c in sorted(counts):
        print(f"    {c} of {len(flags)}          {counts[c]:>4}   "
              f"{100*counts[c]/n:5.1f}%")
    multi = sum(v for k, v in counts.items() if k >= 2)
    print(f"\n    {100*multi/n:.0f}% carry two or more defects at once. That is "
          "what makes this\n    multi-label rather than a set of categories.")
    print()

    # ---------------- per-flag association ----------------
    print("  Does the grouping predict each defect?")
    print()
    alpha_secondary = 0.05 / max(1, len(flags) - 1)
    results = {}
    for k, flag in enumerate(flags):
        a = sum(1 for g, m in scored if g == ga and m[k] == 1)
        na = sum(1 for g, _ in scored if g == ga)
        b = sum(1 for g, m in scored if g == gb and m[k] == 1)
        nb = sum(1 for g, _ in scored if g == gb)
        d, lo, hi = newcombe(a, na, b, nb)
        p = fisher_two_sided(a, na - a, b, nb - b)
        is_primary = flag == primary
        alpha = 0.05 if is_primary else alpha_secondary
        sig = p < alpha
        results[flag] = (d, p, sig, is_primary)
        print(f"    {flag}{'   [PRIMARY]' if is_primary else '   (exploratory)'}")
        print(f"      group {ga}   {100*a/na:5.1f}%   ({a}/{na})")
        print(f"      group {gb}   {100*b/nb:5.1f}%   ({b}/{nb})")
        print(f"      difference  {100*d:+5.1f} pts  [95% CI {100*lo:+.0f} "
              f"to {100*hi:+.0f}]   Fisher p = {p:.4f}")
        print(f"      vs threshold {alpha:.4f}  ->  "
              f"{'SIGNIFICANT' if sig else 'not significant'}")
        print()

    # ---------------- severity ----------------
    xs = [sum(m) for g, m in scored if g == ga]
    ys = [sum(m) for g, m in scored if g == gb]
    d, lo, hi = welch(xs, ys)
    sev = lo > 0 or hi < 0
    print("  Severity -- defects per casting by group:")
    print(f"    group {ga}      {sum(xs)/len(xs):.2f}")
    print(f"    group {gb}      {sum(ys)/len(ys):.2f}")
    print(f"    difference   {d:+.2f}   [95% CI {lo:+.2f} to {hi:+.2f}]   "
          f"{'REAL' if sev else 'not distinguishable'}")
    print(f"    (round 1 measured +0.37 [+0.15 to +0.59] on two broad flags)")
    print()

    # ---------------- verdict ----------------
    hits = [f for f in flags if results[f][2]]
    # direction of EVERY flag, significant or not. If they all lean the same
    # way, the grouping is ranking castings by how damaged they are, and a
    # single flag clearing its threshold does not change that.
    leans = [results[f][0] for f in flags if abs(results[f][0]) > 0.02]
    all_one_way = len(leans) >= 3 and (all(x > 0 for x in leans)
                                       or all(x < 0 for x in leans))
    print("=" * 68)
    if results.get(primary, (0, 1, False, True))[2]:
        print(f"  The primary test PASSED: a narrowly defined '{primary}' is")
        print("  predicted by the grouping. Tightening the definition did the")
        print("  work the broad version could not.")
        if all_one_way and sev:
            print()
            print("  BUT read that with the severity line. All the flags lean")
            print("  the same way and defects-per-casting differs, so the")
            print(f"  grouping may be ranking damage with '{primary}' simply")
            print("  the most visible symptom -- not isolating that defect.")
            print("  A taxonomy would push the flags in OPPOSITE directions.")
            print("  Report it as severity unless you can show otherwise.")
        else:
            print("  That is a defect mode the model learned unsupervised, and")
            print("  a stage-two classifier for it is justified. The flags do")
            print("  not all lean one way, so this is not just severity.")
        if len(hits) > 1:
            print()
            print(f"  Also significant: {', '.join(f for f in hits if f != primary)}"
                  " (exploratory -- treat as")
            print("  hypotheses for a further round, not as results.)")
    elif hits:
        print(f"  The primary test did not pass, but {', '.join(hits)} did as")
        print("  exploratory. Treat that as a hypothesis for the next round,")
        print("  not a result: it was not what this round was set up to test,")
        print("  and promoting it now is the thing pre-registration prevents.")
    elif sev:
        print("  No individual defect is predicted, but severity still is.")
        print("  Tightening the definitions did not rescue a taxonomy, which")
        print("  makes round 1's conclusion stronger rather than weaker: the")
        print("  model encodes HOW BAD a casting is, not WHAT is wrong with")
        print("  it. Two independent measurements, same answer.")
    else:
        print("  Neither individual defects nor severity are predicted here.")
        print("  That disagrees with round 1's severity result, so something")
        print("  differs between the rounds -- most likely the definitions")
        print("  changed what you were counting. Worth reconciling before")
        print("  either result goes in the report.")
    print("=" * 68)
    print()
    if total_defects:
        print(f"  Sample {n} of {total_defects:,} defective castings"
              f"{f'  (groups: {group_sizes})' if group_sizes else ''}.")
    print("  A 20-point difference needs roughly 130 per group for 80% power")
    print("  at these thresholds. Under that, 'not significant' means")
    print("  undetected, not absent.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--score", metavar="CSV")
    ap.add_argument("--data", default="data_clean")
    ap.add_argument("--class-name", default="def_front")
    ap.add_argument("--features", default="defect_types/features.npy")
    ap.add_argument("--assignments", default="defect_types/cluster_assignments.csv")
    ap.add_argument("--out", default="defect_types")
    ap.add_argument("--tag", default="r2",
                    help="round tag; keeps earlier rounds' files intact")
    ap.add_argument("--primary", default="spot",
                    help="the ONE pre-registered flag, tested at 0.05")
    ap.add_argument("--anchor-cluster", type=int, default=2)
    ap.add_argument("--zeros-are-clean", action="store_true",
                    help="read rows marked with 0 as 'no visible defect' "
                         "rather than dropping them as unreadable. Use this "
                         "only when that is what you meant while labelling.")
    ap.add_argument("--from-shortlist", nargs="?", metavar="CSV",
                    const="defect_types/shortlist.csv", default=None,
                    help="label the castings the model is least sure about "
                         "(from apply_defect_types.py) instead of a random "
                         "sample. Worth 2-3x per casting labelled.")
    ap.add_argument("--n", type=int, default=200, help="sample size (even)")
    ap.add_argument("--size", type=int, default=380)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    if args.build and args.score:
        raise SystemExit("--build or --score, not both")
    if args.build:
        return build(args)
    if args.score:
        return score(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
