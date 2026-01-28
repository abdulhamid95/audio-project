# Project: AI Audio Editor Web App (Streamlit)

## Goal
Convert the previous audio processing logic into a **Single-Page Web Application** using **Streamlit**. The app will run locally on the server, allowing me to upload, process, and download files via a browser.

## Tech Stack
* **Frontend/UI:** Streamlit (Pure Python).
* **Audio Processing:** Pydub, FFMpeg.
* **AI Logic:** Faster-Whisper (for repetition detection).
* **Noise Reduction:** Noisereduce.

## Functional Requirements

### 1. The UI (User Interface)
* **Title:** "AI Tutorial Audio Refiner".
* **Upload Widget:** Allow uploading `.mp3`, `.wav`, or `.aup3` files.
* **Settings Sidebar:** Add sliders/checkboxes for:
    * "Remove Silence" (Checkbox + Threshold Slider in dB).
    * "Remove Repetitions" (Checkbox).
    * "Noise Reduction Intensity" (Slider 0.0 to 1.0).

### 2. The Backend Logic (Processing Pipeline)
* **Step 1:** Save the uploaded file to a temporary directory.
* **Step 2 (Conversion):** If the file is `.aup3`, use `ffmpeg` via `subprocess` to convert it to `.wav` first. (Handle errors gracefully if conversion fails).
* **Step 3 (Processing):** Run the cleaning pipeline (Denoise -> Whisper Cut -> Silence Cut).
    * *Important:* Display a **Streamlit progress bar** that updates as each stage completes.
* **Step 4:** Save the processed file as `processed_output.wav`.

### 3. Output
* Show an audio player (`st.audio`) to preview the result.
* Provide a big **Download Button** (`st.download_button`) to save the cleaned file.

## Code Structure
* `app.py`: The main Streamlit application.
* `processor.py`: Contains the logic for noise reduction and whisper cutting (keep logic separate from UI).
* `requirements.txt`: Update to include `streamlit`, `watchdog`.

## Command to Run
Provide the exact command to launch the server (e.g., `streamlit run app.py`).
