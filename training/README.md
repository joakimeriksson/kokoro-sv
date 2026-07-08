# training — reproducible multi-speaker Swedish Kokoro pipeline

The pipeline that produced the published voices: dataset loaders, a common
manifest format, prosody + quality analysis, the training bridge to the
StyleTTS2/kikiri recipe, and the ASR-gated evaluation.

Run from the repo root (this pipeline treats the repo root as the recipe/G2P core).

```bash
# 1. data — streamed from HuggingFace (NST + LibriVox), gender-balanced, quality-gated
python training/scripts/prepare_dataset.py nst --gender Male   --max-hours 5
python training/scripts/prepare_dataset.py nst --gender Female --max-hours 5
python training/scripts/prepare_dataset.py tts_swedish         --max-hours 5
python training/scripts/extract_prosody.py --manifest training/data/manifests/nst.jsonl
python training/scripts/build_mix.py                # training/configs/datasets.yaml

# 2. train the multi-speaker base (smoke first — always)
python training/scripts/train_kokoro.py --manifest training/data/manifests/train_mix.jsonl --name base --smoke --launch
python training/scripts/train_kokoro.py --manifest training/data/manifests/train_mix.jsonl --name base --launch

# 3. evaluate EVERY checkpoint (never pick on loss alone)
python training/scripts/eval_base.py                # ASR-CER + DNSMOS + comb + prosody-responsiveness

# 4. build a HuggingFace voice pack from chosen speakers
python training/scripts/build_hf_pack.py "Signe,Astrid,...,Björn,Sven,..."
```

## Layout

```
training/
  src/kokoro_se/     manifest, Swedish text-normalization, prosody, quality-filter
  scripts/           prepare_dataset, extract_prosody, build_mix, train_kokoro,
                     eval_base, evaluate_samples, voice_gallery, build_hf_pack, ...
  configs/           datasets.yaml (mix), testset_sv.txt, voice_names.json
  docs/              PLAN.md
```

## Principles (baked into the scripts)

1. **One compute job at a time** — via `../gpu_run.sh` (kernel `flock`). Unified
   CPU+GPU memory boxes freeze if two heavy jobs run at once.
2. **Never rank a checkpoint on one metric** — ASR-CER (intelligibility) → DNSMOS
   (quality) → artifact metrics. A comb-optimized checkpoint was unintelligible.
3. **Validate the corpus against downstream filters** before bulk generation.
4. **Prosody lives in the style vector** — train on many real speakers so the
   predictor learns to use it; single-speaker distillation is flat.

Full narrative + measurements: [`../docs/PLAN.md`](../docs/PLAN.md) and `../docs/`.
