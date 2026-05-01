#!/usr/bin/env python3
"""Push-to-toggle local Dutch transcription with faster-whisper.

Default hotkey: Windows+Shift
- First press: start microphone recording
- Second press: stop recording, transcribe, and type text at current cursor
"""

from __future__ import annotations

import argparse
import tempfile
import threading
import time
from pathlib import Path

import keyboard
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record with a global hotkey and type Dutch transcript at cursor."
    )
    parser.add_argument("--model", default="small", help="faster-whisper model size/path")
    parser.add_argument("--samplerate", type=int, default=16000, help="Recording sample rate")
    parser.add_argument("--channels", type=int, default=1, help="Audio channels (1 = mono)")
    parser.add_argument("--hotkey", default="windows+shift", help="Global hotkey to toggle recording")
    return parser.parse_args()


class HotkeyTranscriber:
    def __init__(self, model_name: str, samplerate: int, channels: int):
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        self.samplerate = samplerate
        self.channels = channels

        self._is_recording = False
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ANN001
        if status:
            print(f"Audio status: {status}")
        with self._lock:
            if self._is_recording:
                self._frames.append(indata.copy())

    def toggle(self) -> None:
        with self._lock:
            should_start = not self._is_recording
        if should_start:
            self.start_recording()
        else:
            self.stop_and_transcribe()

    def start_recording(self) -> None:
        with self._lock:
            self._frames.clear()
            self._is_recording = True

        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        print("🔴 Recording started... Press hotkey again to stop.")

    def stop_and_transcribe(self) -> None:
        with self._lock:
            self._is_recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            if not self._frames:
                print("No audio captured.")
                return
            audio = np.concatenate(self._frames, axis=0)

        if audio.size == 0:
            print("No audio captured.")
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        sf.write(tmp_path, audio, self.samplerate)

        try:
            print("Processing...")
            segments, _ = self.model.transcribe(
                str(tmp_path),
                language="nl",
                vad_filter=True,
                beam_size=1,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()

            if not text:
                print("Done (empty transcript).")
                return

            keyboard.write(text)
            print("Done")
        finally:
            tmp_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()

    print(f"Loading faster-whisper model '{args.model}' (CPU int8)...")
    transcriber = HotkeyTranscriber(
        model_name=args.model,
        samplerate=args.samplerate,
        channels=args.channels,
    )

    keyboard.add_hotkey(args.hotkey, transcriber.toggle)
    print(f"Ready. Press {args.hotkey} to start/stop recording. Press Ctrl+C to quit.")

    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    main()
