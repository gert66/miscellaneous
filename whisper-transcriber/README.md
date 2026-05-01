# Whisper Transcriber (lokaal + hotkey)

Deze versie draait lokaal en werkt met een globale hotkey:

- **1e keer hotkey**: opname start
- **2e keer hotkey**: opname stopt, transcriptie start, tekst wordt direct getypt waar je cursor staat

## 1) Installatie

```bash
python -m pip install faster-whisper sounddevice soundfile keyboard numpy
```

> Let op: `ffmpeg` moet op je systeem beschikbaar zijn.

## 2) Gebruik

```bash
python hotkey_transcriber.py --model small --hotkey windows+shift
```

Handige opties:

- `--model`: modelnaam of pad voor faster-whisper (standaard: `small`)
- `--hotkey`: standaard `windows+shift`
- `--samplerate`: standaard `16000`
- `--channels`: standaard `1`

## 3) Gedrag

- Transcriptie gebeurt **alleen** nadat opname stopt.
- Taal is geforceerd op Nederlands (`nl`).
- Voor CPU gebruikt het script `compute_type="int8"` voor snelheid.
- Bij stoppen zie je duidelijk: `Processing...` en daarna `Done`.

## 4) Windows tips

- Start terminal als Administrator als globale hotkeys niet werken.
- Zorg dat je microfoonrechten aan staan in Windows Privacy instellingen.
- In sommige apps kan beveiligde invoer geautomatiseerd typen blokkeren.
