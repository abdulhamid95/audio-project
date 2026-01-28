# Project: AI Tutorial Audio Studio (Streamlit Web App)

## Role
Act as a Senior Python AI Engineer. Build a production-ready, local web application using **Streamlit** that automates the post-production of audio tutorials.

## The Goal
Create a web interface where I can upload raw audio recordings (including `.aup3`, `.wav`, `.mp3`). The system must process the audio to remove noise, cut silences, and **intelligently remove repeated phrases and stutters** using OpenAI's Whisper model.

## Tech Stack
* **Frontend:** `streamlit`
* **Audio Processing:** `pydub`, `scipy`
* **AI/Transcription:** `faster-whisper` (CPU/GPU friendly)
* **Noise Reduction:** `noisereduce`
* **Matching Logic:** `difflib` (Standard Python)
* **System Tools:** `ffmpeg` (via `subprocess` for conversion)

---

## Architecture & Logic (Files to Create)

### 1. `requirements.txt`
Include: `streamlit`, `pydub`, `faster-whisper`, `noisereduce`, `scipy`, `numpy`.

### 2. `utils.py` (Helper Functions)
* **Function: `convert_input(file_path)`**
    * Handle `.aup3`, `.mp3`, `.wav`.
    * Since `.aup3` is complex, attempt to convert it to `.wav` using a `subprocess.run(['ffmpeg', '-i', ...])` command.
    * If `.aup3` conversion fails (due to codec issues), return a clear error asking the user to export as WAV from Audacity.
    * Return path to a standardized `.wav` file.

### 3. `processor.py` (Core Logic - The Brain)
This file must contain the class `AudioProcessor` with the following pipeline:

#### Step A: Noise Reduction
* Load audio.
* Use `noisereduce` to profile the background noise (using the first 0.5s as noise sample if possible, or stationary mode) and strip it.

#### Step B: Smart Repetition Removal (CRITICAL)
* **Algorithm:** "The Last Take Strategy".
* **Process:**
    1.  Transcribe audio using `faster-whisper` to get `segments` (text + start/end timestamps).
    2.  Iterate through segments comparing `Segment[i]` vs `Segment[i+1]`.
    3.  **Detection Logic:**
        * **Condition 1 (Repetition):** If `SequenceMatcher(a, b).ratio() > 0.85` (Fuzzy Match).
        * **Condition 2 (False Start):** If `Segment[i].text` is a distinct substring of the *start* of `Segment[i+1].text` (e.g., "In this..." vs "In this video we will").
    4.  **Action:** If either condition is met, **mark `Segment[i]` (the first one) for DELETION**. Assume the second attempt is the correct one.
    5.  **Reconstruction:** Create a new audio timeline excluding the deletion ranges.

#### Step C: Silence Removal
* After removing repetitions, scan the new audio.
* Detect silence chunks > 800ms (threshold -40dB).
* Truncate them to 100ms (keep pacing natural, don't just delete).

### 4. `app.py` (The User Interface)
* **Page Config:** Title "AI Audio Refiner".
* **Sidebar Controls:**
    * Checkbox: "Remove Noise"
    * Checkbox: "Remove Repetitions" (Default: True)
    * Slider: "Silence Threshold (dB)" (Default: -40)
* **Main Area:**
    * **File Uploader:** Accept `.aup3`, `.wav`, `.mp3`.
    * **Process Button:** When clicked:
        1.  Show `st.status` or progress bar.
        2.  Run the pipeline.
        3.  Log actions to UI (e.g., "Found repetition: 'Hello...' -> Removed").
    * **Results:**
        * `st.audio` player to listen to the result.
        * `st.download_button` to download the final cleaned WAV file.

---

## Implementation Instructions for the Agent
1.  **Do not use placeholders.** Write full, working code.
2.  **Error Handling:** Wrap the `.aup3` conversion in a try/except block. If ffmpeg fails, instruct the user via `st.error` to upload a WAV.
3.  **Performance:** Load the Whisper model only once (cache it using `@st.cache_resource`).

**START IMPLEMENTING NOW.**
