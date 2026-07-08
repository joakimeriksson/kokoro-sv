"""kokoro-sv CLI.

    kokoro-sv voices
    kokoro-sv speak "Hej, jag är CandyTron!" --voice Stina --out hej.wav
    kokoro-sv blend "God morgon!" --a Björn --b Nils --mix 0.7 --out mix.wav
"""
from __future__ import annotations

import argparse


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kokoro-sv", description="Swedish Kokoro TTS")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("voices", help="list available voices")

    sp = sub.add_parser("speak", help="synthesize text")
    sp.add_argument("text")
    sp.add_argument("--voice", default="Stina")
    sp.add_argument("--out", default="out.wav")
    sp.add_argument("--speed", type=float, default=1.0)

    bl = sub.add_parser("blend", help="synthesize with a blend of two voices")
    bl.add_argument("text")
    bl.add_argument("--a", required=True)
    bl.add_argument("--b", required=True)
    bl.add_argument("--mix", type=float, default=0.5, help="weight of --a (0..1)")
    bl.add_argument("--out", default="out.wav")

    args = ap.parse_args(argv)
    from .core import SwedishKokoro

    if args.cmd == "voices":
        print("\n".join(SwedishKokoro().voices))
        return
    tts = SwedishKokoro()
    if args.cmd == "speak":
        print("wrote", tts.speak(args.text, args.voice, args.out, args.speed))
    elif args.cmd == "blend":
        print("wrote", tts.speak(args.text, tts.blend(args.a, args.b, args.mix), args.out))


if __name__ == "__main__":
    main()
