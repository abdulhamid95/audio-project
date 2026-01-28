# AI Tutorial Audio Refiner

A Streamlit web application that automatically cleans up tutorial recordings using AI-powered audio processing.

## Features

- **Noise Reduction** - Removes background static and hiss
- **Smart Repetition Removal** - Uses AI (Whisper) to detect and remove repeated phrases, keeping only the final take
- **Silence Trimming** - Trims long pauses to improve pacing

## Requirements

- Python 3.10+
- FFmpeg (must be installed on your system)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/abdulhamid95/audio-project.git
cd audio-project
```

2. Create and activate a virtual environment:
```bash
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install FFmpeg (if not already installed):
```bash
# Ubuntu/Debian:
sudo apt install ffmpeg

# macOS:
brew install ffmpeg

# Windows:
# Download from https://ffmpeg.org/download.html and add to PATH
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at http://localhost:8501

### Processing Options

- **Remove Long Silences** - Trims silences longer than 800ms
- **Silence Threshold (dB)** - Adjust what volume level is considered silence (-60 to -20 dB)
- **Remove Repetitions** - AI-powered detection and removal of repeated phrases
- **Noise Reduction Intensity** - Control strength of noise removal (0.0 to 1.0)

### Supported Formats

- MP3
- WAV
- Audacity Project (.aup3)

## Project Structure

```
├── app.py              # Streamlit web interface
├── processor.py        # Audio processing pipeline
├── main.py             # CLI tool (alternative to web UI)
├── audio_editor/       # Core audio processing modules
│   ├── converter.py    # Format conversion (FFmpeg)
│   ├── denoiser.py     # Noise reduction
│   ├── transcriber.py  # Whisper transcription
│   ├── repetition.py   # Repetition detection/removal
│   └── silence.py      # Silence detection/trimming
└── requirements.txt
```

## CLI Usage (Alternative)

You can also use the command-line tool:
```bash
python main.py recording.wav -o cleaned.wav
python main.py recording.wav --skip-denoise --similarity 0.85
```

Run `python main.py --help` for all options.
