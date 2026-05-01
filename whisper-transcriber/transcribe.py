#!/usr/bin/env python3
"""Local audio transcription with OpenAI Whisper.

Usage:
  python transcribe.py audio.wav --model small --language nl --output transcript.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import whisper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files locally with Whisper (no monthly word limits)."
    )
    parser.add_argument("audio", type=Path, help="Path to an audio/video file")
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model to use (tiny, base, small, medium, large)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force input language (e.g. 'nl', 'en'). Autodetect when omitted.",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Transcribe in source language or translate to English.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output .txt file. Prints to stdout when omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.audio.exists():
        raise FileNotFoundError(f"Bestand niet gevonden: {args.audio}")

    print(f"Loading model '{args.model}'...")
    model = whisper.load_model(args.model)

    print(f"Transcribing {args.audio}...")
    result = model.transcribe(
        str(args.audio),
        language=args.language,
        task=args.task,
        fp16=False,
    )

    transcript = result["text"].strip()

    if args.output:
        args.output.write_text(transcript + "\n", encoding="utf-8")
        print(f"Saved transcript to {args.output}")
    else:
        print("\n--- Transcript ---\n")
        print(transcript)


if __name__ == "__main__":
    main()
