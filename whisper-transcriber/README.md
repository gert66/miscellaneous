# Whisper Transcriber (lokaal + hotkey)

Deze versie draait lokaal en werkt met een globale hotkey:

- **1e keer hotkey**: opname start
- **2e keer hotkey**: opname stopt, Whisper transcribeert, tekst wordt direct getypt waar je cursor staat

## 1) Installatie

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install openai-whisper sounddevice soundfile keyboard numpy
```

> Let op: `ffmpeg` moet op je systeem beschikbaar zijn.

## 2) Gebruik

```bash
python hotkey_transcriber.py --model small --language nl --hotkey windows+shift
```

Handige opties:

- `--model`: `tiny`, `base`, `small`, `medium`, `large`
- `--language`: optioneel (bijv. `nl`, `en`)
- `--task`: `transcribe` (standaard) of `translate`
- `--hotkey`: standaard `windows+shift`
- `--samplerate`: standaard `16000`

## 3) Windows tips

- Start terminal als Administrator als globale hotkeys niet werken.
- Zorg dat je microfoonrechten aan staan in Windows Privacy instellingen.
- In sommige apps kan beveiligde invoer geautomatiseerd typen blokkeren.
