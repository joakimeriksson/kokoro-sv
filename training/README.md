# kokoro-se — svensk Kokoro-träningspipeline med prosodifokus

Reproducerbar pipeline för att träna/fintuna Kokoro-82M för svenska med mer
naturlig och dynamisk prosodi. Bygger vidare på `../swedish-kokoro` (tränings-
recept, neural G2P, inferens med notch-kedja) och RUN2-lärdomarna.

## MVP-status

| # | Del | Status |
|---|-----|--------|
| 1 | Projektstruktur | ✅ |
| 2 | Manifestformat (`src/kokoro_se/dataset_manifest.py`) | ✅ |
| 3 | Svensk textnormalisering (siffror/datum/tider/förkortn.) | ✅ |
| 4 | Audio-konvertering (mono 24 kHz, loudness, 200 ms svanspaddning) | ✅ |
| 5 | NST-loader (streamar HF `KTH/nst` parquet, ingen 44 GB-nedladdning) | ✅ |
| 6 | CSS10-loader (lokal Kaggle-zip) | ✅ |
| 7 | Dataset-mix (`configs/datasets.yaml` + prosodi-översampling) | ✅ |
| 8 | Fast svenskt testset (`configs/testset_sv.txt`) | ✅ |
| 9 | Checkpoint-jämförelseprover + kvalitetsbatteri | ✅ |
| — | RixVox, Common Voice, active learning, träningsscript | ⏳ efter MVP |

## Snabbstart

```bash
cd kokoro-se
uv venv && uv pip install -e ".[datasets]"
kokoro-se normalize "Mötet är 2026-07-06 kl. 14:30 och kostar 1250 kr."
kokoro-se prepare --dataset nst --max-hours 8
kokoro-se prepare --dataset css10 --raw data/raw/css10_sv
kokoro-se extract-prosody --manifest data/manifests/nst.jsonl
kokoro-se build-mix
kokoro-se eval --checkpoint <ckpt.pth> --voicepack <vp.pt> --name my-experiment
```

Träning sker tills vidare via `../swedish-kokoro/train_gb10.sh` (kikiri-receptet)
med filelists genererade från `data/manifests/train_mix.jsonl` — ett dedikerat
`train_kokoro.py` som gör den kopplingen automatiskt är nästa steg efter MVP.

## Hårda regler (dyrköpta)

1. **En beräkningsprocess åt gången på GB10.** Minnet är enhetligt (CPU+GPU
   delar 128 GB); två samtidiga tunga jobb har hårdfryst maskinen två gånger.
   ALLT tungt körs genom `../gpu_run.sh` (kernel-flock). Inga undantag,
   inte ens "lätt CPU-jobb".
2. **Ranka aldrig checkpoints på en ensam metrik.** Kvalitetsbatteriet gäller:
   ASR-CER (begriplighet, hård gate) → DNSMOS (perceptuell) → artefaktmått.
   RUN2:s "bästa" komb-checkpoint var obegriplig rappakalja.
3. **Optimera inte bara loss** (plan §13). `evaluate_samples.py` körs på varje
   checkpoint; välj med öronen bland batteri-godkända kandidater.
4. **Validera korpus mot nedströmsfilter INNAN bulkarbete** — 35 % av RUN2:s
   första korpus var >150 tecken och blev bortkastad GPU-tid (12 s-taket).
5. **Prosodi bor i stilvektorn.** Voicepack = [128 timbre | 128 prosodi];
   medelvärdesbildning över många klipp plattar melodin. Bygg voicepacks från
   uttrycksfulla klipp, eller skarva prosodihalvan från skådespelat material.

## Datakällor

| Källa | Roll | Prio |
|---|---|---|
| NST/KTH (HF `KTH/nst`) | fonetik, uttal, många talare | hög |
| CSS10 Swedish | single-speaker-referens | hög |
| Common Voice sv | variation (hårt kvalitetsfilter) | medel |
| RixVox | prosodivariation | experimentell |
| Egen inspelning (1–2 h) | speaker adaptation | sist |
