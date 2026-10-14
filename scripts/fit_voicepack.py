"""Fit a Kokoro voice pack to a target speaker WITHOUT retraining the model.

A Kokoro voice is a style tensor [510, 1, 256] (one 256-d style vector per input
length). This searches that space on the FROZEN Swedish model for the pack whose
output sounds most like a target speaker, given ~200+ clips of that speaker
(e.g. a VoxCPM2 distillation dataset):

    pack = sum_k softmax(logits)_k * P_k  +  delta

P_k are existing packs (the 10 Swedish voices + Kokoro-82M's own 54), so the fit
starts on the model's natural style manifold; delta is a small learned 256-d
offset (L2-regularized) for what no blend reaches. Because the model is not
touched, every existing voice is unaffected and the pack drops straight into the
voice server.

Objective (gradient through the decoder via the undecorated forward):
  1 - cos(ECAPA(generated), ECAPA centroid of the target clips)
  + w_ltas * || mean log-mel(generated) - mean log-mel(target) ||^2
  + w_delta * ||delta||^2

Guards against an adversarial vector that fools ECAPA but sounds wrong:
  * evaluation uses an INDEPENDENT speaker model (x-vector) never optimized on,
  * Whisper CER on held-out Swedish sentences,
  * the nearest single existing voice is reported as the baseline.

  python scripts/fit_voicepack.py --target data-gen/dataset_vox_female/wavs \\
      --out data-gen/fitted_voices/vox_female --steps 300
Run in the kokoro-voice-server venv (kokoro, speechbrain, faster-whisper, kokoro-sv).
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import re
import sys
import time
import warnings

import numpy as np
import soundfile as sf
import torch
import torchaudio

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

SR = 24000
HELD_OUT = [
    "Hej och välkommen! Vill du ha en godisbit idag?",
    "Varsågod, här kommer en Dumle till dig. Ha en riktigt fin dag!",
    "Jag flyttar Riesen från ruta A2 till ruta O0 nu.",
    "Ursäkta, jag hörde inte riktigt. Kan du säga det en gång till?",
    "Roboten plockar godiset försiktigt och lägger det framför dig.",
    "Visste du att Sverige äter mer lösgodis än nästan alla andra länder?",
]


def median_f0(x: np.ndarray, sr: int = 16000) -> float:
    fr, hop = int(0.04 * sr), int(0.01 * sr); f = []
    for i in range(0, len(x) - fr, hop):
        seg = x[i:i + fr] - x[i:i + fr].mean()
        if np.sqrt(np.mean(seg ** 2)) < 0.02:
            continue
        ac = np.correlate(seg, seg, "full")[fr - 1:]
        lo, hi = int(sr / 400), int(sr / 70)
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.45 * ac[0]:
            f.append(sr / k)
    return float(np.median(f)) if f else 0.0


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def load_g2p():
    import kokoro_sv
    vend = os.path.join(os.path.dirname(kokoro_sv.__file__), "_vendor")
    sys.path.insert(0, vend)
    os.environ.setdefault("SV_NEURAL_G2P", "nst_g2p")
    os.environ.setdefault("SV_G2P_DIR", os.path.join(vend, "g2p"))
    from g2p_sv import SwedishG2P
    return SwedishG2P(backend="neural")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="dir of target-speaker wavs")
    ap.add_argument("--texts", default="data-gen/corpus_vox_sv.txt", help="training sentences")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--max-target", type=int, default=300)
    ap.add_argument("--w-ltas", type=float, default=0.5)
    ap.add_argument("--w-delta", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    from huggingface_hub import hf_hub_download, list_repo_files
    from kokoro import KModel
    from speechbrain.inference.speaker import EncoderClassifier
    from faster_whisper import WhisperModel

    model = KModel(repo_id="hexgrad/Kokoro-82M",
                   config=hf_hub_download("Joakim/kokoro-sv-voices", "config.json"),
                   model=hf_hub_download("Joakim/kokoro-sv-voices", "kokoro_sv.pth")).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    fwd = type(model).forward_with_tokens.__wrapped__   # bypass @torch.no_grad

    sv_files = sorted(f for f in list_repo_files("Joakim/kokoro-sv-voices")
                      if f.startswith("voices/") and f.endswith(".pt"))
    base_files = sorted(f for f in list_repo_files("hexgrad/Kokoro-82M")
                        if f.startswith("voices/") and f.endswith(".pt"))
    names = [f[7:-3] for f in sv_files] + ["base:" + f[7:-3] for f in base_files]
    P = torch.stack([torch.load(hf_hub_download("Joakim/kokoro-sv-voices", f), weights_only=True)
                     for f in sv_files] +
                    [torch.load(hf_hub_download("hexgrad/Kokoro-82M", f), weights_only=True)
                     for f in base_files]).float()                 # [K, 510, 1, 256]
    log(f"basis: {len(names)} packs ({len(sv_files)} Swedish)")

    cache = os.path.expanduser("~/.cache/speechbrain")
    ecapa = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                           savedir=f"{cache}/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})
    xvect = EncoderClassifier.from_hparams("speechbrain/spkrec-xvect-voxceleb",
                                           savedir=f"{cache}/spkrec-xvect-voxceleb", run_opts={"device": "cpu"})
    for enc in (ecapa, xvect):
        for p in enc.mods.parameters():
            p.requires_grad_(False)
    asr = WhisperModel("small", device="cpu", compute_type="int8")
    mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=512, hop_length=160, n_mels=64)

    def embed(enc, a16: torch.Tensor) -> torch.Tensor:
        feats = enc.mods.compute_features(a16)
        feats = enc.mods.mean_var_norm(feats, torch.ones(a16.shape[0]))
        e = enc.mods.embedding_model(feats, torch.ones(a16.shape[0])).reshape(a16.shape[0], -1)
        return torch.nn.functional.normalize(e, dim=-1)

    def logmel_mean(a16: torch.Tensor) -> torch.Tensor:
        return torch.log(mel(a16) + 1e-5).mean(-1).reshape(-1)

    # --- target statistics ---------------------------------------------------
    files = sorted(glob.glob(os.path.join(args.target, "*.wav")))
    rng.shuffle(files)
    files = files[: args.max_target]
    te, tx, tl = [], [], []
    with torch.no_grad():
        for f in files:
            x, sr = sf.read(f, dtype="float32")
            a = torchaudio.functional.resample(torch.from_numpy(x)[None], sr, 16000)
            te.append(embed(ecapa, a)[0]); tx.append(embed(xvect, a)[0]); tl.append(logmel_mean(a))
    t_ecapa = torch.nn.functional.normalize(torch.stack(te).mean(0), dim=0)
    t_xvect = torch.nn.functional.normalize(torch.stack(tx).mean(0), dim=0)
    t_ltas = torch.stack(tl).mean(0)
    # how similar the target's own clips are to its centroid = the ceiling
    ceil_e = float(torch.stack([e @ t_ecapa for e in te]).median())
    ceil_x = float(torch.stack([e @ t_xvect for e in tx]).median())
    def to16(path: str) -> np.ndarray:
        x, sr = sf.read(path, dtype="float32")
        return np.interp(np.arange(0, len(x), sr / 16000), np.arange(len(x)), x).astype(np.float32)
    t_f0 = float(np.median([median_f0(to16(f)) for f in files[:40]]))
    log(f"target: {len(files)} clips | own-centroid similarity: ecapa {ceil_e:.3f}, xvect {ceil_x:.3f} "
        f"| median F0 {t_f0:.0f} Hz")

    # --- text -> token ids ---------------------------------------------------
    g2p = load_g2p()

    def ids_of(text: str) -> list[int]:
        ipa = g2p(text).replace("ʏ", "y")
        return [i for i in (model.vocab.get(c) for c in ipa) if i is not None]

    corpus = [l.strip() for l in open(args.texts, encoding="utf-8") if l.strip()]
    corpus = [t for t in corpus if 4 <= len(t.split()) <= 16]
    rng.shuffle(corpus)
    train = [ids for ids in (ids_of(t) for t in corpus[:240]) if 8 <= len(ids) <= 300]
    held = [(t, ids_of(t)) for t in HELD_OUT]
    log(f"train sentences: {len(train)}, held-out: {len(held)}")

    def render(ids: list[int], pack_rows: torch.Tensor) -> torch.Tensor:
        ref = pack_rows[len(ids) - 1]                     # [1, 256]
        audio, _ = fwd(model, torch.LongTensor([[0, *ids, 0]]), ref, 1.0)
        return audio.reshape(1, -1)

    def build_pack(logits: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(logits, 0)
        return (w[:, None, None, None] * P).sum(0) + delta[None, None, :]   # [510, 1, 256]

    def norm_text(s: str) -> str:
        return " ".join(re.sub(r"[^\w ]", "", s.lower()).split())

    def cer(r: str, h: str) -> float:
        r, h = norm_text(r), norm_text(h)
        d = list(range(len(h) + 1))
        for i, a in enumerate(r, 1):
            p, d[0] = d[0], i
            for j, b in enumerate(h, 1):
                p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a != b))
        return d[len(h)] / max(1, len(r))


    def evaluate(pack: torch.Tensor, tag: str, save_audio: bool = False) -> dict:
        es, xs, cs, fs = [], [], [], []
        with torch.no_grad():
            for k, (text, ids) in enumerate(held):
                a24 = render(ids, pack)
                a16 = torchaudio.functional.resample(a24, SR, 16000)
                es.append(float(embed(ecapa, a16)[0] @ t_ecapa))
                xs.append(float(embed(xvect, a16)[0] @ t_xvect))
                segs, _ = asr.transcribe(a16[0].numpy(), language="sv", beam_size=1)
                cs.append(cer(text, " ".join(s.text for s in segs)))
                fs.append(median_f0(a16[0].numpy()))
                if save_audio:
                    sf.write(os.path.join(args.out, f"{tag}_{k}.wav"), a24[0].numpy(), SR)
        return {"ecapa": float(np.mean(es)), "xvect": float(np.mean(xs)), "cer": float(np.mean(cs)),
                "f0": float(np.median(fs))}

    # --- baseline: nearest single existing voice, and warm start --------------
    probe = train[0]
    sims = []
    with torch.no_grad():
        for k in range(len(names)):
            a16 = torchaudio.functional.resample(render(probe, P[k]), SR, 16000)
            sims.append(float(embed(ecapa, a16)[0] @ t_ecapa))
    order = np.argsort(sims)[::-1]
    log("nearest existing voices: " + ", ".join(f"{names[i]} {sims[i]:.2f}" for i in order[:5]))
    nearest = int(order[0])
    base_eval = evaluate(P[nearest], "baseline_" + names[nearest].replace(":", "_"), save_audio=True)
    log(f"baseline {names[nearest]}: {base_eval}")

    logits = torch.tensor(np.array(sims) * 20.0, dtype=torch.float32)   # soft warm start
    logits = (logits - logits.max()).requires_grad_(True)
    delta = torch.zeros(256, requires_grad=True)
    opt = torch.optim.Adam([{"params": [logits], "lr": 0.05}, {"params": [delta], "lr": 0.01}])

    history = [{"step": 0, "baseline": names[nearest], **base_eval}]
    best = None
    t0 = time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad()
        pack = build_pack(logits, delta)
        loss_val = 0.0
        for ids in rng.sample(train, args.batch):
            a16 = torchaudio.functional.resample(render(ids, pack), SR, 16000)
            l_spk = 1 - embed(ecapa, a16)[0] @ t_ecapa
            l_ltas = ((logmel_mean(a16) - t_ltas) ** 2).mean()
            loss = (l_spk + args.w_ltas * l_ltas) / args.batch
            loss.backward(retain_graph=True)
            loss_val += float(loss)
        (args.w_delta * (delta ** 2).sum()).backward()
        opt.step()
        if step % 10 == 0:
            w = torch.softmax(logits.detach(), 0)
            top = torch.topk(w, 3)
            mix = ", ".join(f"{names[i]} {float(v):.2f}" for v, i in zip(top.values, top.indices))
            log(f"step {step}/{args.steps} loss {loss_val:.4f} |delta| {float(delta.norm()):.2f} "
                f"| mix {mix} | {(time.time()-t0)/step:.1f}s/step")
        if step % args.eval_every == 0 or step == args.steps:
            pack = build_pack(logits.detach(), delta.detach())
            ev = evaluate(pack, f"step{step:04d}", save_audio=(step == args.steps))
            ev["step"] = step
            history.append(ev)
            torch.save(pack.clone(), os.path.join(args.out, f"step{step:04d}.pt"))
            log(f"eval step {step}: {ev}")
            # Pick on the INDEPENDENT x-vector score, traded against intelligibility:
            # late steps gain ~0.001 x-vector (noise) while CER can triple. Score =
            # xvect - 0.1 * CER, among fits within the intelligibility guard.
            ev["score"] = ev["xvect"] - 0.1 * ev["cer"]
            if ev["cer"] <= base_eval["cer"] + 0.05 and (best is None or ev["score"] > best["score"]):
                best = ev
    json.dump({"names": names, "history": history, "best": best,
               "target_ceiling": {"ecapa": ceil_e, "xvect": ceil_x}},
              open(os.path.join(args.out, "fit.json"), "w"), indent=2)
    if best:
        import shutil
        shutil.copy(os.path.join(args.out, f"step{best['step']:04d}.pt"), os.path.join(args.out, "best.pt"))
        ev = evaluate(torch.load(os.path.join(args.out, "best.pt")), "best", save_audio=True)
        log(f"BEST step {best['step']}: {ev}  (baseline {names[nearest]}: {base_eval})")
    else:
        log("no fit beat the intelligibility guard; keeping nothing")


if __name__ == "__main__":
    main()
