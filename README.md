# AI Audio Refiner

A Streamlit web application that automatically cleans up tutorial recordings using AI-powered audio processing.

## Features

- **Noise Reduction** - Removes background static/hiss using the first 0.5s as noise profile
- **Smart Repetition Removal** - Uses Whisper AI with "The Last Take Strategy":
  - Detects repeated phrases (>85% similarity)
  - Detects false starts (incomplete sentences followed by complete ones)
  - Always keeps the last take, removes earlier attempts
- **Silence Trimming** - Truncates silences >800ms down to 100ms for natural pacing

## Requirements

- Python 3.10+
- FFmpeg

## Installation

```bash
# Clone the repository
git clone https://github.com/abdulhamid95/audio-project.git
cd audio-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Install FFmpeg

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Usage

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Processing Options (Sidebar)

| Option | Default | Description |
|--------|---------|-------------|
| Remove Noise | On | Apply noise reduction |
| Remove Repetitions | On | AI-powered repetition detection |
| Silence Threshold | -40 dB | Audio below this is silence |

### Supported Formats

- WAV
- MP3
- Audacity Project (.aup3) - requires FFmpeg codec support

## Project Structure

```
├── app.py           # Streamlit UI
├── processor.py     # AudioProcessor class (core logic)
├── utils.py         # File conversion helpers
├── main.py          # CLI alternative
├── audio_editor/    # Legacy module (can be removed)
└── requirements.txt
```

## How Repetition Detection Works

The "Last Take Strategy" algorithm:

1. Transcribe audio with Whisper to get timestamped segments
2. Compare consecutive segments:
   - **Repetition**: `SequenceMatcher(a, b).ratio() > 0.85`
   - **False Start**: Segment A is a prefix of Segment B
3. Mark the FIRST segment for deletion (keep the last take)
4. Reconstruct audio excluding deleted ranges

## CLI Usage

```bash
python main.py recording.wav -o cleaned.wav
python main.py recording.wav --skip-denoise --similarity 0.85
python main.py --help
```
