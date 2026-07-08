"""kokoro-se CLI — thin wrappers over scripts/ so everything has one entrypoint.

    kokoro-se prepare --dataset nst --max-hours 8
    kokoro-se extract-prosody --manifest data/manifests/nst.jsonl
    kokoro-se build-mix --config configs/datasets.yaml
    kokoro-se eval --checkpoint outputs/checkpoints/latest.pth --voicepack vp.pt
    kokoro-se normalize "Mötet är 2026-07-06 kl. 14:30 och kostar 1 250 kr."
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[2]


def _run(script: str, *args: str):
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], check=True)


@click.group()
def main():
    """Swedish Kokoro training pipeline."""


@main.command()
@click.option("--dataset", type=click.Choice(["nst", "css10"]), required=True)
@click.option("--max-hours", default=8.0)
@click.option("--raw", default=None)
def prepare(dataset, max_hours, raw):
    """Convert a raw dataset to the common manifest + 24 kHz wavs."""
    args = [dataset, "--max-hours", str(max_hours)]
    if raw:
        args += ["--raw", raw]
    _run("prepare_dataset.py", *args)


@main.command("extract-prosody")
@click.option("--manifest", required=True)
def extract_prosody_cmd(manifest):
    """Add prosody features to every manifest entry (in place)."""
    _run("extract_prosody.py", "--manifest", manifest)


@main.command("build-mix")
@click.option("--config", default="configs/datasets.yaml")
@click.option("--out", default="data/manifests/train_mix.jsonl")
def build_mix(config, out):
    """Build a weighted training manifest from dataset manifests."""
    _run("build_mix.py", "--config", config, "--out", out)


@main.command()
@click.option("--checkpoint", required=True)
@click.option("--voicepack", required=True)
@click.option("--name", default=None)
def eval(checkpoint, voicepack, name):
    """Render the fixed testset from a checkpoint + battery-score it."""
    args = ["--checkpoint", checkpoint, "--voicepack", voicepack]
    if name:
        args += ["--name", name]
    _run("evaluate_samples.py", *args)


@main.command()
@click.argument("text")
def normalize(text):
    """Show Swedish text normalization for TEXT."""
    from kokoro_se.text_normalization import normalize_text_sv
    click.echo(normalize_text_sv(text))


if __name__ == "__main__":
    main()
