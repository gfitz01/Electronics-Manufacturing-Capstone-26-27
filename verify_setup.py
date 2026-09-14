"""Check the environment before you spend an afternoon training.

Run from the repo root, with the venv active:

    python verify_setup.py

Every check prints PASS, FAIL or SKIP with a reason. Nothing here trains
anything or touches your data; the whole script should finish in well under a
minute.
"""

import sys
import traceback
from pathlib import Path

# Allow running as  python capstone-setup/verify_setup.py  from the repo root.
REPO_ROOT = Path(__file__).resolve().parent
if not (REPO_ROOT / "ManufacturingNet").is_dir():
    REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

RESULTS = []


def check(name):
    """Decorator: run a check, catch anything, record the outcome."""
    def wrapper(fn):
        print(f"\n--- {name} ---")
        try:
            outcome = fn()
        except Exception:
            traceback.print_exc()
            RESULTS.append((name, "FAIL", "raised an exception (above)"))
            return fn
        if outcome is None:
            outcome = ("PASS", "")
        status, detail = outcome
        RESULTS.append((name, status, detail))
        print(f"{status}: {detail}" if detail else status)
        return fn
    return wrapper


@check("Python version")
def _python():
    major, minor = sys.version_info[:2]
    detail = f"{sys.version.split()[0]} at {sys.executable}"
    print(detail)
    if major != 3 or minor < 9:
        return ("FAIL", f"need Python 3.9-3.12, found {major}.{minor}")
    if minor > 12:
        return ("PASS", f"{major}.{minor} is newer than tested (3.9-3.12)")
    return ("PASS", detail)


@check("Virtual environment active")
def _venv():
    active = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if not active:
        return ("FAIL", "no venv active -- run .\\.venv\\Scripts\\activate")
    return ("PASS", sys.prefix)


@check("torch and torchvision")
def _torch():
    import torch
    import torchvision
    detail = f"torch {torch.__version__}, torchvision {torchvision.__version__}"
    print(detail)

    # A mismatched pair is the single most common broken install.
    t_major_minor = torch.__version__.split("+")[0].rsplit(".", 1)[0]
    print(f"torch base version: {t_major_minor}")
    return ("PASS", detail)


@check("GPU availability")
def _gpu():
    import torch
    if not torch.cuda.is_available():
        return ("SKIP", "no CUDA -- training will run on CPU (slow but correct)")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # Actually move a tensor, don't just trust the flag.
    x = torch.randn(64, 64, device="cuda")
    _ = (x @ x).sum().item()
    return ("PASS", f"{name}, {total:.1f} GB, matmul on device OK")


@check("float32 pipeline (the speed fix)")
def _dtype():
    import torch
    from torchvision import transforms
    from PIL import Image

    # Reproduce the transform pipeline the models build.
    pipeline = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    tensor = pipeline(Image.new("RGB", (300, 300)))
    if tensor.dtype != torch.float32:
        return ("FAIL", f"expected float32, got {tensor.dtype}")
    return ("PASS", f"{tuple(tensor.shape)} {tensor.dtype}")


@check("InterpolationMode accepted (the torchvision fix)")
def _interpolation():
    from torchvision import transforms
    transforms.Resize((224, 224),
                      interpolation=transforms.InterpolationMode.BILINEAR)
    # The old integer form should now be gone from the source.
    hits = []
    for path in (REPO_ROOT / "ManufacturingNet").rglob("*.py"):
        if "interpolation=2" in path.read_text(encoding="utf-8",
                                               errors="ignore"):
            hits.append(path.name)
    if hits:
        return ("FAIL", f"interpolation=2 still present in {', '.join(hits)} "
                        "-- is the patch applied?")
    return ("PASS", "no integer interpolation left in the package")


@check("pretrained stem preserved (the accuracy fix)")
def _stem():
    import torch
    from torchvision.models import alexnet, AlexNet_Weights
    from ManufacturingNet.models._backbone_utils import adapt_first_conv

    model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    original = model.features[0]
    original_weight = original.weight.clone()

    # 3-channel input: the layer must come back untouched, weights intact.
    same = adapt_first_conv(original, 3)
    if same is not original:
        return ("FAIL", "3-channel input rebuilt the layer unnecessarily")
    if not torch.equal(same.weight, original_weight):
        return ("FAIL", "pretrained weights were modified")

    # 1-channel input: rebuilt, but geometry preserved.
    adapted = adapt_first_conv(original, 1)
    if adapted.kernel_size != original.kernel_size:
        return ("FAIL", f"kernel changed {original.kernel_size} -> "
                        f"{adapted.kernel_size}")
    if adapted.stride != original.stride:
        return ("FAIL", f"stride changed {original.stride} -> {adapted.stride}")

    return ("PASS", f"kernel {original.kernel_size} stride {original.stride} "
                    "preserved; pretrained weights kept")


@check("Library imports")
def _imports():
    from ManufacturingNet.models import AlexNet, ResNet, VGG  # noqa: F401
    from ManufacturingNet.datasets import (prepare_image_folder,  # noqa: F401
                                           use_local_archive, count_images)
    from ManufacturingNet.reporting import RunReport  # noqa: F401
    return ("PASS", "models, datasets and reporting all import")


@check("Forward and backward pass")
def _forward():
    import torch
    import torch.nn as nn
    from torchvision.models import alexnet
    from ManufacturingNet.models._backbone_utils import adapt_first_conv

    model = alexnet(weights=None)
    model.features[0] = adapt_first_conv(model.features[0], 3)
    model.classifier[-1] = nn.Linear(4096, 2)
    model = model.float()

    batch = torch.randn(2, 3, 224, 224)
    target = torch.tensor([0, 1])
    output = model(batch)
    if output.shape != (2, 2):
        return ("FAIL", f"expected (2, 2) logits, got {tuple(output.shape)}")

    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    if model.features[0].weight.grad is None:
        return ("FAIL", "no gradient reached the first conv layer")
    return ("PASS", f"logits {tuple(output.shape)}, loss {loss.item():.4f}, "
                    "gradients flow")


@check("Run report output")
def _report():
    import json
    import tempfile
    from ManufacturingNet.reporting import RunReport

    report = RunReport(model_name="VerifyNet",
                       class_names=["ok", "defective"])
    report.log_epoch(0, train_loss=0.5, train_accuracy=70.0,
                     val_loss=0.6, val_accuracy=65.0)
    report.set_predictions([0, 0, 0, 1, 1], [0, 0, 1, 1, 1])

    with tempfile.TemporaryDirectory() as tmp:
        paths = report.save(tmp)
        document = json.loads(Path(paths["results_json"]).read_text())

    accuracy = document["final_metrics"]["accuracy_pct"]
    if accuracy != 80.0:
        return ("FAIL", f"expected 80.0% accuracy, got {accuracy}")
    summary = document["final_metrics"]["inspection_summary"]
    if summary["actual_damaged_count"] != 2:
        return ("FAIL", "inspection summary counted defects wrong")
    return ("PASS", f"results.json written, accuracy {accuracy}%, "
                    f"damaged {summary['actual_damaged_pct']}%")


def main():
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    width = max(len(name) for name, _, _ in RESULTS)
    failed = 0
    for name, status, detail in RESULTS:
        print(f"{status:<5} {name:<{width}}  {detail}")
        if status == "FAIL":
            failed += 1

    print("=" * 60)
    if failed:
        print(f"{failed} check(s) failed. Fix these before training.")
        return 1
    print("All checks passed. You're clear to run a small training job.")
    print("Next: Step 6 in SETUP.md -- get the images onto disk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
