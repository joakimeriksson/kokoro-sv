# swedish-kokoro

Train a **native Swedish voice for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)**
(StyleTTS2 architecture) — from raw data to a fast, deployable multi-speaker model
with dynamic prosody and named voices. This repo is **code + downloader scripts
only**; every weight, dataset, and audio file is `.gitignore`d and fetched or
regenerated on demand. Trained voices are published to HuggingFace.

## What's here

| result | |
|---|---|
| **Multi-speaker Swedish base** | one model, 22+ real Swedish speakers, **prosody-responsive** (the style vector genuinely steers delivery: 4.9→9.5 semitone range on demand) |
| **10-voice pack** | 5 female + 5 male named voicepacks (512 KB each), interpolatable |
| **Neural Swedish G2P** | hybrid NST-lexicon + transformer, with a growing tech/brand lexicon (MQTT, RISE, YOLO, …) |
| **Reproducible pipeline** | dataset loaders, manifest format, prosody QC battery, training bridge, evaluation |

Voices are distilled/trained from **CC0 sources** (NST + Swedish LibriVox) and
published on HF. See the model cards there.

## Repository layout

```
.                     recipe core: G2P, configs, inference (synth_real.py), weight conversion
  g2p/                neural Swedish G2P (code; model+lexicon auto-download from HF)
  pipeline/           reproducible multi-speaker pipeline (was: kokoro-se)
    src/kokoro_se/    manifest, text-normalization, prosody, quality-filter
    scripts/          download/prepare data, train, evaluate, build voice packs
    configs/          dataset mix, test set, voice names, training configs
  data-gen/           Chatterbox teacher — synthesize single-speaker data (was: swedish-chatterbox)
  docs/               training-recipe.md (detailed), RUN1.md, PLAN.md
  gpu_run.sh          flock wrapper — ONE compute job at a time (see Lessons)
```

## Reproduce, end to end

Nothing but code is committed; these steps download/regenerate everything.

```bash
# 0. base weights + kikiri recipe + PL-BERT
bash setup_3090.sh                       # (works on the GB10 too; adjust venv per docs)

# 1. data — streamed from HuggingFace, gender-balanced, quality-gated
cd pipeline
python scripts/prepare_dataset.py nst          --gender Male   --max-hours 5
python scripts/prepare_dataset.py nst          --gender Female --max-hours 5
python scripts/prepare_dataset.py tts_swedish                  --max-hours 5
python scripts/extract_prosody.py --manifest data/manifests/nst.jsonl
python scripts/build_mix.py                     # configs/datasets.yaml

# 2. train the multi-speaker base (StyleTTS2 via kikiri; smoke first!)
python scripts/train_kokoro.py --manifest data/manifests/train_mix.jsonl --name base --smoke --launch
python scripts/train_kokoro.py --manifest data/manifests/train_mix.jsonl --name base --launch

# 3. evaluate EVERY checkpoint (never pick on loss alone)
python scripts/eval_base.py                     # ASR-CER + DNSMOS + comb + prosody-responsiveness

# 4. build a HuggingFace voice pack from chosen speakers
python scripts/build_hf_pack.py "Signe,Astrid,…,Björn,Sven,…"
```

Alternatively, `data-gen/` synthesizes a clean single-speaker corpus with
Chatterbox (the RUN1 distillation path) — see `data-gen/README.md`.

## Hard-won lessons (baked into the code)

1. **One compute job at a time.** The GB10 has unified CPU+GPU memory; two heavy
   jobs hard-froze it. Everything runs through `gpu_run.sh` (kernel `flock`).
2. **Never rank a checkpoint on one metric.** Gate on ASR-CER (intelligibility) →
   DNSMOS (quality) → artifact metrics. A comb-optimized checkpoint was unintelligible.
3. **Validate the corpus against downstream filters** before bulk generation
   (35 % of an early corpus exceeded the 12 s training cap = wasted GPU).
4. **Prosody lives in the style vector** ([128 timbre | 128 prosody]); train on many
   real speakers so the predictor learns to *use* it — single-speaker distillation is flat.
5. **The GAN/adversarial stage is needed** for decoder quality (with enough data);
   low decoder-lr prevents the fine-tune upsampler-tone artifacts (notched at inference).

## Credits

Built on [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0),
[yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) (MIT), and
[semidark/kikiri-tts](https://github.com/semidark/kikiri-tts). Data: NST
([Språkbanken](https://www.nb.no/sprakbanken/), CC0) and Swedish LibriVox
([TTS-Swedish](https://huggingface.co/datasets/datadriven-company/TTS-Swedish), CC0).
Evaluation: [KBLab VoxRex](https://huggingface.co/KBLab/wav2vec2-large-voxrex-swedish)
+ Microsoft DNSMOS. Data manufactured with [Chatterbox](https://github.com/resemble-ai/chatterbox) (MIT).
