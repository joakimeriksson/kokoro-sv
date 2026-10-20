# Plan: two new Swedish Kokoro voices distilled from VoxCPM2

Status as of 15 Oct 2026. Written so the work can resume after a context reset.

## Goal

Add one female and one male voice to the Swedish Kokoro model (next to Stina,
Greta and the other eight), distilled from two VoxCPM2 takes Joakim picked by ear
in the "Greta vs VoxCPM2" trial (artifact, 13 Sep 2026, run on the DGX Spark).

## Decisions taken

| Decision | Choice | Why |
|---|---|---|
| Female reference | trial take *plain, 10 steps* of the "tretton sorters godis" line | chosen by ear; median F0 ~183 Hz |
| Male reference | trial take *design, 10 steps* of the same line | chosen by ear; median F0 ~118 Hz |
| Cloning mode | VoxCPM2 **ultimate cloning**: `reference_wav_path` + `prompt_wav_path` + `prompt_text` | best identity (ECAPA 0.79–0.90 to the take) |
| Style instruction ("happier") | **none** | `(glad, entusiastisk...)` made voices livelier but drifted identity (0.69–0.81); Joakim preferred ultimate by ear |
| Numbers in generation text | keep **digits** ("Candytron 4000") | VoxCPM2 reads digits well; spelled-out "fyratusen" came out as "fyra- tusen" |
| Starting point for training | the **Swedish multi-speaker base**, not plain Kokoro | Swedish already learned; single-speaker distillation from plain Kokoro is flat (training/README principle 4) |
| Training data mix | new voices as speakers 11–12 **plus** a slice of the original mix | all packs share one model; training on new voices alone degrades the existing ten |

## Facts established

- VoxCPM2 plain and design modes invent a **new speaker every call**: the four
  trial takes of the same line are four different people (ECAPA 0.33–0.51). The
  recordings themselves are the voices. They are saved in
  `data-gen/refs/vox_female_ref.wav`, `vox_male_ref.wav` (+ `vox_refs.txt`),
  committed in `faa19f8` with a `.gitignore` exception (`*.wav` is ignored).
  They are decoded from the artifact's MP3s; the Spark may still hold the
  original 48 kHz WAVs, which would be better references.
- Prompt text of both takes: *"Det finns tretton sorters godis på bordet, till
  exempel Geisha, Marianne och Plopp. Säg vilken du vill ha, så plockar jag den
  åt dig."*
- VoxCPM2 (voxcpm 2.0.3) runs on the MacBook (MPS, float32) at RTF ~1.0; about
  20 clips/min on corpus lines. License: Apache 2.0 per the VoxCPM docs (confirm
  the repo LICENSE file before publishing packs).
- Corpus `data-gen/corpus_vox_sv.txt`: the 200 CandyTron lines first, then 2101
  filtered NST sentences (`KTH/nst` via the HF datasets server). The old
  `datadriven-company/TTS-Swedish` source is no longer accessible.
- Pilot: 20 clips per voice, all on-voice (min ECAPA 0.74 / 0.77; max 0.43 to the
  other voice).

### Forgetting: the Swedish fine-tune already changed the other languages

The voice server runs en/fr/es/it through the **same** fine-tuned weights
(`KPipeline(..., model=self._model)`). Same sentences and voices, original
Kokoro-82M vs `kokoro_sv.pth`, scored with Whisper small:

| lang | CER base → Swedish | heard as that language (Whisper LID) |
|---|---|---|
| en | 0.000 → 0.000 | 1.00 → 0.99 |
| fr | 0.007 → **0.121** | 1.00 → 0.97 |
| es | 0.000 → 0.014 | 0.98 → **0.80** |
| it | 0.012 → 0.019 | 0.95 → **0.66** |

English is untouched; French lost intelligibility; Spanish and Italian sound
less like themselves. More Swedish training would push this further.

**Planned fix (independent of the new voices):** serve non-Swedish languages
from the original Kokoro-82M weights in kokoro-voice-server (a second `KModel`,
~330 MB RAM). Swedish keeps `kokoro_sv.pth`; other languages become immune to
any future Swedish training. Re-run the measurement afterwards (script:
`scripts/forgetting_vs_base.py`, run in the kokoro-voice-server venv).

## Steps

1. **Generate datasets** — RUNNING on the Mac since 15 Oct ~00:30.
   `data-gen/batch_voxcpm.py`, female then male in one background job, log in
   `data-gen/voxgen.log`. ~2 h per voice, ~2.5 h audio each. Output
   `data-gen/dataset_vox_female/`, `dataset_vox_male/` (wavs 24 kHz +
   `speakers.csv`, same contract as the Chatterbox `batch_synth.py`). Resumable:
   rerun the same command. Stop with `pkill -f batch_voxcpm`.
   Command (per voice):
   `PYTORCH_ENABLE_MPS_FALLBACK=1 <voxcpm-venv>/bin/python batch_voxcpm.py --ref refs/vox_female_ref.wav --speaker sv_vox_female --out dataset_vox_female --device mps`
2. **Verify datasets**: ECAPA similarity per clip vs reference (drop < 0.7),
   ASR CER gate (existing `verify_dataset.py` / quality filter), durations.
3. **Get to the training box** (Spark GB10 or the 3090): BLOCKED on its address.
   Not in `~/.ssh/config`, shell history or the repo. Copy datasets there; also
   fetch the original trial WAVs if present.
4. **Quick test without training**: extract style vectors for the two new
   speakers from the existing multi-speaker base checkpoint (needs the style
   encoder, which only the training checkpoint has; the published
   `kokoro_sv.pth` drops it). Listen.
5. **Continued fine-tune** of the Swedish base: add the two datasets as new
   speakers to the training mix (`training/scripts/build_mix.py`,
   `training/configs/datasets.yaml`) with a slice of the original data, short
   run, smoke first, one GPU job at a time (`gpu_run.sh`).
6. **Evaluate every checkpoint** (`training/scripts/eval_base.py`: ASR-CER →
   DNSMOS → artifacts) for the new voices AND the existing ten.
7. **Build the pack**: `training/scripts/build_hf_pack.py` with the ten existing
   names plus two new ones; upload to `Joakim/kokoro-sv-voices` after Joakim
   approves by ear. Add names to `face/languages.toml` `tts_voices` in mcp-agents.
8. **Voice server**: DONE 15 Oct (kokoro-voice-server `--other-langs-weights base`,
   default). Non-Swedish languages now run on original Kokoro-82M weights; measured
   end to end: en/fr/it CER 0.000/0.007/0.000, heard-as 0.97-0.99, latency
   unchanged (~0.26 s). Swedish training can no longer degrade them, so
   rehearsal data in step 5 is optional (only needed for a single-model deploy).

## Open items

- Training box address (Spark or 3090) — needed for steps 3–7.
- Name for the male voice (female is Klara).
- Original 48 kHz trial WAVs on the Spark, if still there.
- `faa19f8` in swedish-kokoro is committed locally, not pushed.

## Results, night of 15 Oct 2026 (training box still unreachable)

Instead of retraining, each voice's style pack was **fitted on the frozen
Swedish model** (`scripts/fit_voicepack.py`: blend of the 64 existing packs +
small offset, gradient through Kokoro against the target's ECAPA centroid, mean
log-mel and — added after the first male fit came out 6 semitones high —
Kokoro's own predicted F0). Existing voices are untouched by construction.

Datasets (`scripts/verify_vox_dataset.py`, Whisper small): female kept 2259/2301
(2.67 h), male 2256/2301 (2.67 h); median similarity to the take 0.87 / 0.88,
median CER 0.010 / 0.012.

Packs (female named Klara by Joakim 15 Oct; male name provisional), served via `--extra-voices data-gen/fitted_voices/extra`:

| | similarity to take | pitch (take) | server-Whisper CER 1.0x / 1.3x |
|---|---|---|---|
| Klara (female, step 100) | 0.81 | 198 Hz (183) | 0.075 / 0.082 |
| Viktor (male, F0 fit, step 200) | 0.74 | 116 Hz (118) | 0.097 / 0.100 |
| Nils (existing) | 0.36 to male take | 148 Hz | 0.054 / 0.054 |
| Greta (existing default) | 0.35 to female take | 254 Hz | 0.092 / 0.143 |

The first male fit (timbre only) is kept as
`fitted_voices/vox_male/Viktor_night.pt` (0.64, 165 Hz).

Next: Joakim listens and names them; upload to `Joakim/kokoro-sv-voices`
after approval. Full retraining (steps 3–7) remains the way to go beyond
~0.8 similarity, and needs the training box.

## Retraining per voice (Oct 2026)

Fitted packs only move timbre; prosody lives in the shared predictor, so a fitted
voice is as flat as the base. `training/scripts/finetune_voice.py` continues stage 2
from the multi-speaker base on one voice's verified set -> a separate model
(published kokoro_sv.pth and its ten voices untouched).

- **Klara** (Spark, Oct 15-16, 5 epochs, 2209 clips): F0-std 6.51 -> 7.7 st (target
  7.10), median F0 197 -> 172 Hz (target 158), CER 0.027 -> 0.023-0.025, DNSMOS
  3.11 -> 3.41-3.48. Candidates e00003 / e00004. Not yet exported or on the laptop.
- **Viktor**: queued. `data-gen/at_home_viktor.sh` waits for the Spark on the home
  LAN, ships `dataset_vox_male` + `training/scripts/voice_job.sh`, and starts
  smoke -> 5-epoch fine-tune -> eval (~4 h). Log: `finetune_viktor.log`,
  `eval_viktor.log` on the Spark.
- Then: export inference weights for the chosen epochs, serve each voice with its
  own model in kokoro-voice-server, listen, decide on HF upload.

### Oct 19: new male take
The first male take (~118 Hz; clips median 109 Hz) sounded too low by ear, so a
retrain on it would inherit that. New take picked by Joakim from VoxCPM2 voice
design candidates (`data-gen/design_takes.py`, `refs/male_candidates/`): #8,
"A Swedish man with a youthful, slightly high-pitched voice, playful and
enthusiastic", ~128 Hz -> `refs/vox_male2_ref.wav`. `data-gen/male2_pipeline.sh`
generates `dataset_vox_male2` (~95 min on the Mac), verifies it, then runs
`at_home_viktor.sh` on that set. The old `vox_male_ref.wav` / `dataset_vox_male` are kept.

### Oct 19: retrained Klara approved by ear
Joakim: "retrained is great!" (the fitted packs: "not so great"). Klara is now
served from her own weights: `fitted_voices/extra/Klara.pth` (e00004 export from
`training/outputs/eval/klara/e00004/kokoro_sv.pth`) + `Klara.pt` (`vp_e00004.pt`).
The fitted packs moved to `fitted_voices/retired/`. Viktor is served by nobody until
his retrain (dataset_vox_male2) lands; requests for him fall back to the default voice.
Gotcha fixed in kokoro-voice-server 5230cad: StyleTTS2 exports have "module." keys and
parametrized weight_norm; KModel's non-strict load left the decoder random (pure noise).

### Oct 20: Viktor retrained (dataset_vox_male2, take 8)
Spark, 5 epochs, 07:40-11:42; eval 11:45. Target clips: F0-std 6.85 st, 131 Hz.

| ckpt | F0-std | F0 | CER | DNSMOS |
|---|---|---|---|---|
| base + pack | 5.58 | 151 | 0.024 | 3.17 |
| e00000 | 6.90 | 143 | 0.031 | 3.41 |
| e00001 | 7.05 | 148 | 0.025 | 3.38 |
| e00002 | 6.64 | 150 | 0.026 | 3.40 |
| e00003 | 6.52 | 147 | 0.023 | 3.38 |
| e00004 | 6.72 | 149 | 0.027 | 3.36 |

Served for listening: `extra/Viktor.*` = e00001 (liveliest), `extra/ViktorE5.*` = e00004
(Klara's epoch). Pick by ear, then drop the other and publish Klara + Viktor together.
