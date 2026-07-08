"""Render the fixed Swedish testset from a checkpoint and battery-score it.

Wraps the proven swedish-kokoro inference path (synth_real.py: KModel conversion,
neural G2P, EOS-trim, notch chain) and the RUN2 eval battery. GPU-lock enforced.

    python scripts/evaluate_samples.py --checkpoint <ckpt.pth> --voicepack <vp.pt> \
        [--name kokoro-se-base-e3]

Outputs to outputs/samples/<name>/ + a battery line in outputs/reports/eval.log.

RUN2 principle (plan §13): never pick a checkpoint on loss alone — this script
exists so every checkpoint gets EARS + battery. Compare renders, then choose.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SK = ROOT.parent          # sibling repo with the proven pipeline
GPU_LOCK = ROOT.parent / "gpu_run.sh"        # ONE compute job at a time (GB10 rule)
TESTSET = ROOT / "configs" / "testset_sv.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--voicepack", required=True)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    name = args.name or Path(args.checkpoint).stem
    out = ROOT / "outputs" / "samples" / name
    py = SK / "recipe" / ".venv" / "bin" / "python"

    render = [str(GPU_LOCK), str(py), str(SK / "synth_real.py"),
              "--checkpoint", args.checkpoint, "--voicepack", args.voicepack,
              "--texts", str(TESTSET), "--out-dir", str(out)]
    subprocess.run(render, check=True, cwd=SK,
                   env={"SV_NEURAL_G2P": "nst_g2p", "PATH": "/usr/bin:/bin"})

    battery = subprocess.run(
        [str(GPU_LOCK), str(py), str(SK / "eval_renders.py"),
         "--texts", str(TESTSET), str(out)],
        cwd=SK, capture_output=True, text=True)
    line = battery.stdout.strip().splitlines()[-1] if battery.stdout.strip() else "battery failed"
    report = ROOT / "outputs" / "reports" / "eval.log"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("a", encoding="utf-8") as f:
        f.write(f"{name}\t{line}\n")
    print(line)
    print(f"samples -> {out}\nreport  -> {report}")


if __name__ == "__main__":
    main()
