# Plan: Dynamisk svensk Kokoro via multi-speaker-träning

## Varför (uppmätt, inte gissat)

RUN2-mätningar (F0-variabilitet i semitoner, naturligt tal ≈ 6–7.5 st):

| Källa | F0-std |
|---|---|
| NST-människa (018 / 98) | 7.4 / 6.0 |
| Chatterbox-lärare (018 / 98) | 7.3 / 5.3 |
| Våra Kokoro-röster (018 / 98) | 5.9 / 3.3–4.5 |

Två flaskhalsar: (1) Chatterbox ärver referensklippets neutrala uppläsningsprosodi
— i ALLA språk, vid ALLA exaggeration-inställningar (24-varianters svep, platt);
(2) medelvärdes-voicepack + deterministisk prosodiprediktor tränad på smal
prosodifördelning ⇒ regression mot monotoni. Fixen för båda: träna på MÅNGA
riktiga talare med RIKTIG prosodivariation, så att stilmanifolden blir bred och
prediktorn lär sig ANVÄNDA prosodidimensionerna. Sedan är röstidentitet bara en
voicepack (Kokoros design) och känslolägen hamnar inom manifolden.

## Fas 0 — Publicera nuvarande röster på Hugging Face (startpunkter/baseline)

Artefakter per röst: checkpoint (KModel-format), voicepack, notch-kedjans
parametrar, Swedish config/vocab, testset-renderingar, modellkort med hela
provenansen och mätvärdena.

- Licenser: NST = CC0, Chatterbox = MIT ⇒ rösterna kan publiceras öppet.
  **OBS: mood-packs härleder prosodi ur RAVDESS (CC BY-NC-SA) — publiceras
  separat med NC-flagga eller utelämnas.**
- Repo-förslag: `<user>/kokoro-sv-female-018`, `<user>/kokoro-sv-male-98`
  (+ ev. samlings-repo `kokoro-sv-baseline`).
- Modellkortet är ärligt: "prosodiskt plana baseline-röster; v2 med
  multi-speaker-bas kommer" + länk till kikiri-tts, StyleTTS2, Kokoro-82M.
- GATE: användarens godkännande innan upload (utåtriktat, hans HF-konto).

## Fas 1 — Datagrund: flera källor, riktig prosodi

> **REVIDERAD (2026-07-07) till "Fas 1 lite":** coverage slår volym. ~12 h räcker
> för manifold-breddning (SLM-golvet är ~1 500 klipp; RUN2 gav perfekt CER på
> 3,5 h). Mindre data = ~1 h/epok = experiment över en natt istället för dagar.
> Starta med NST (~6 h, ~30 talare) + TTS-Swedish/LibriVox (~5 h, 9 talare),
> översampla frågande/expressiv. CSS10, Common Voice och RixVox aktiveras BARA
> om prosodiresponstestet (Fas 2) visar ett uppmätt gap. Full lista nedan är
> eskaleringsmenyn, inte startplanen.

1. **NST full**: `python training/scripts/prepare_dataset.py --dataset nst --max-hours 20` (streamning
   funkar, verifierad). ~100 talare, neutral men äkta prosodi. Kvalitetsbatteri på allt.
2. **CSS10 Swedish** (~10 h, en talare, berättande audiobook-prosodi): ladda ner
   Kaggle-zip, `prepare --dataset css10`.
3. **Common Voice sv**: loader byggs; HÅRT filter (batteri: CER-gate mot
   transkript + DNSMOS ≥ 3.0) — behåll kanske 10–20 %.
4. **RixVox (riksdagen)**: loader byggs; spontant, argumenterande, känslosamt tal
   = prosodiguldet. Experimentell: segmentera + batteri + prosodiklassning.
4b. **TTS-Swedish / LibriVox (datadriven-company)**: 40 h, 9 talare, 24 kHz,
   transkriberad + DNSMOS-skattad — kurerad LibriVox, verktyg finns redan
   (swedish-kokoro prepare_data-formatet). Bästa bulk-källan för äkta
   audiobok-prosodi. Innehåller "Jordens inre"-läsaren (skånska) — bra som
   träningsdata; hennes RIKTIGA röst kan sedan bli en voicepack (med skånskan).
5. **Prosodirapport**: `extract-prosody` på alla manifest; rapport per källa
   (F0-std-fördelning, klasser, talhastighet) → styr mixvikterna med data.

Mål: ≥ 30 h batteri-godkänt, ≥ 50 talare, prosodiklasser täckta (frågande,
expressiv, långsam, snabb — inte bara neutral).

## Fas 2 — Multi-speaker basträning ("kokoro-sv-base")

1. **Smoke först** (RUN2-disciplin): `multispeaker: true` i StyleTTS2-configen är
   OPRÖVAT i vår kikiri-miljö — 1 epok på 50 klipp × 5 talare; finita losses,
   inga NaN. Om det strular: undersök kikiris multispeaker-stöd innan fullskala.
2. `train_kokoro.py`-bryggan: train_mix.jsonl → G2P → StyleTTS2-filelists med
   numeriska talar-ID (RUN2-gotcha) + configgenerering.
3. Full träning: stage 1 + joint stage 2 från epok 0 (mel-GAN + SLM — degenererar
   inte med >1 500 klipp), max_len 800, ft_lr 1e-5, batteri per epok EFTER körning
   (aldrig samtidigt — GPU-låset).
4. **Prosodiresponstest** (nytt, avgörande): rendera testsetet med voicepacks
   extraherade från talare med hög resp. låg F0-varians. Mät spridningen i
   utsignalen. Responsen = hur mycket stilvektorn styr prosodin. Basen är lyckad
   när frågeintonation stiger, utrop lyfter och lugna talares packs saktar tempot.

## Fas 3 — Prosodifintuning ("kokoro-sv-prosody")

- Urval via manifest: prosodiklasserna frågande/expressiv/energisk översamplade,
  RixVox-segment med hög F0-varians, långa meningar med naturliga pauser.
- Kort fintuning (2–4 epoker) ovanpå basen; batteri + prosodiresponstest per epok.
- Mål: renderad F0-std ≥ naturlig baseline (≈ 7 st) med bibehållen CER ≤ 3 %.

## Fas 4 — Talaradaptering (CandyTron-rösterna, v2)

Nu är rösterna bara voicepacks — och vi kan välja källa:

- **Alternativ A (ingen ny data): riktiga NST 018/98.** ~312 klipp ≈ 25 min var
  finns redan i korpusen. Ovanpå en stark bas räcker det ofta för adaptering —
  och då är Chatterbox HELT ute ur kedjan (ingen lärar-flathet, ingen distillering).
- **Alternativ B: egen inspelning 1–2 h** (bokläsning + frågor + utrop + siffror
  + teknikord) för en "kokoro-sv-joakim"-röst. Active-learning-verktyget (§10 i
  huvudplanen) genererar inspelningspaketen.
- Mood-packs v3: extrahera om känslopaketen mot den breda basen — bör nu ligga
  inom manifolden (ingen DNSMOS-dipp).

## Fas 5 — Publicera v2 + uppströms

- HF: v2-röster ersätter/kompletterar baseline-repos, med prosodimätningar i korten.
- kikiri-tts: discussion-#8-post + små PR:ar (transformers-5-fixen, ARM-noter,
  svenska receptdokument). Vårt repo refererar deras som pinnad submodule.

## Framgångsmått (mäts, inte känns)

| Mått | Baseline (nu) | Mål |
|---|---|---|
| F0-std, renderad kvinnlig | 5.9 st | ≥ 7.0 st |
| F0-std, renderad manlig | 3.3–4.5 st | ≥ 5.5 st |
| Frågeintonation (F0-slut minus F0-mitt på "?"-meningar) | omätt | tydligt positiv |
| CER (batteri) | 1–2 % | ≤ 3 % |
| DNSMOS | 3.3 | ≥ 3.2 |
| Mood-pack-separation (lugn vs exalterad tempo/F0) | svag–medel | tydlig, utan DNSMOS-dipp |

Ordningsregler: GPU-låset gäller allt; batteri + öron på varje checkpoint;
korpus valideras mot 12 s-taket INNAN träning; inga beslut på enstaka metriker.
