"""Core audio processing logic - The Brain."""

import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable

import numpy as np
from scipy.io import wavfile
import noisereduce as nr
from pydub import AudioSegment
from faster_whisper import WhisperModel


@dataclass
class Segment:
    """A transcribed segment with timing."""
    text: str
    start: float
    end: float


@dataclass
class DeletionRange:
    """A time range marked for deletion."""
    start: float
    end: float
    reason: str


class AudioProcessor:
    """
    Audio processing pipeline implementing "The Last Take Strategy".

    Pipeline:
    A. Noise Reduction
    B. Smart Repetition Removal
    C. Silence Removal
    """

    def __init__(
        self,
        whisper_model: WhisperModel,
        log_callback: Callable[[str], None] | None = None
    ):
        """
        Initialize the processor.

        Args:
            whisper_model: Pre-loaded Whisper model instance.
            log_callback: Function to call with log messages.
        """
        self.model = whisper_model
        self.log = log_callback or (lambda x: None)

    def process(
        self,
        input_path: str,
        output_path: str,
        remove_noise: bool = True,
        remove_repetitions: bool = True,
        silence_threshold_db: float = -40.0,
        progress_callback: Callable[[float], None] | None = None,
    ) -> str:
        """
        Run the complete processing pipeline.

        Args:
            input_path: Path to input WAV file.
            output_path: Path for output WAV file.
            remove_noise: Whether to apply noise reduction.
            remove_repetitions: Whether to detect and remove repetitions.
            silence_threshold_db: Threshold for silence detection.
            progress_callback: Function(progress: float) for updates.

        Returns:
            Path to processed audio file.
        """
        update = progress_callback or (lambda x: None)
        current_path = input_path

        # Step A: Noise Reduction (0-30%)
        if remove_noise:
            self.log("Applying noise reduction...")
            update(0.1)
            denoised_path = output_path.replace(".wav", "_denoised.wav")
            current_path = self._reduce_noise(current_path, denoised_path)
            self.log("Noise reduction complete.")
            update(0.3)

        # Step B: Repetition Removal (30-70%)
        if remove_repetitions:
            self.log("Transcribing audio with Whisper AI...")
            update(0.35)
            segments = self._transcribe(current_path)
            self.log(f"Found {len(segments)} speech segments.")
            update(0.5)

            self.log("Analyzing for repetitions and false starts...")
            deletions = self._detect_repetitions(segments)

            if deletions:
                self.log(f"Found {len(deletions)} sections to remove:")
                for d in deletions:
                    self.log(f"  - {d.reason}")

                no_rep_path = output_path.replace(".wav", "_norep.wav")
                current_path = self._remove_segments(current_path, no_rep_path, deletions)
                self.log("Repetitions removed.")
            else:
                self.log("No repetitions detected.")
            update(0.7)

        # Step C: Silence Removal (70-100%)
        self.log("Detecting and trimming silences...")
        update(0.75)
        current_path = self._trim_silences(
            current_path,
            output_path,
            threshold_db=silence_threshold_db
        )
        self.log("Silence trimming complete.")
        update(1.0)

        return output_path

    def _reduce_noise(
        self,
        input_path: str,
        output_path: str,
        noise_sample_seconds: float = 0.5
    ) -> str:
        """Apply noise reduction using noisereduce."""
        sample_rate, audio_data = wavfile.read(input_path)

        # Convert to float
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_float = audio_data.astype(np.float32) / 2147483648.0
        else:
            audio_float = audio_data.astype(np.float32)

        # Handle stereo
        if len(audio_float.shape) > 1:
            audio_float = np.mean(audio_float, axis=1)

        # Use first N seconds as noise profile
        noise_samples = int(noise_sample_seconds * sample_rate)
        noise_clip = audio_float[:noise_samples]

        # Apply noise reduction
        reduced = nr.reduce_noise(
            y=audio_float,
            sr=sample_rate,
            y_noise=noise_clip,
            prop_decrease=0.8,
            stationary=True,
        )

        # Normalize
        max_val = np.max(np.abs(reduced))
        if max_val > 0:
            reduced = reduced / max_val * 0.95

        # Save
        reduced_int = (reduced * 32767).astype(np.int16)
        wavfile.write(output_path, sample_rate, reduced_int)

        return output_path

    def _transcribe(self, audio_path: str) -> list[Segment]:
        """Transcribe audio using Whisper."""
        segments_gen, _ = self.model.transcribe(
            audio_path,
            word_timestamps=True,
            vad_filter=True,
        )

        segments = []
        for seg in segments_gen:
            text = seg.text.strip()
            if text:
                start = seg.words[0].start if seg.words else seg.start
                end = seg.words[-1].end if seg.words else seg.end
                segments.append(Segment(text=text, start=start, end=end))

        return segments

    def _detect_repetitions(
        self,
        segments: list[Segment],
        similarity_threshold: float = 0.85
    ) -> list[DeletionRange]:
        """
        Detect repetitions using "The Last Take Strategy".

        Conditions for deletion:
        1. Fuzzy match: SequenceMatcher ratio > 0.85
        2. False start: Segment[i] is substring of start of Segment[i+1]

        The FIRST segment is marked for deletion (keep the last take).
        """
        if len(segments) < 2:
            return []

        deletions = []

        for i in range(len(segments) - 1):
            curr = segments[i]
            next_seg = segments[i + 1]

            curr_text = curr.text.lower().strip()
            next_text = next_seg.text.lower().strip()

            should_delete = False
            reason = ""

            # Condition 1: Fuzzy match (repetition)
            ratio = SequenceMatcher(None, curr_text, next_text).ratio()
            if ratio > similarity_threshold:
                should_delete = True
                reason = f"Repetition ({ratio:.0%} similar): '{curr.text[:40]}...' -> Removed"

            # Condition 2: False start (substring at beginning)
            elif len(curr_text) > 3 and next_text.startswith(curr_text[:len(curr_text)]):
                should_delete = True
                reason = f"False start: '{curr.text[:30]}...' -> Removed"

            # Also check if current is a short prefix of next (common stutter pattern)
            elif len(curr_text) < len(next_text) * 0.5:
                # Check if current text words appear at start of next
                curr_words = curr_text.split()
                next_words = next_text.split()
                if len(curr_words) >= 1 and len(next_words) >= len(curr_words):
                    if curr_words == next_words[:len(curr_words)]:
                        should_delete = True
                        reason = f"Incomplete phrase: '{curr.text[:30]}...' -> Removed"

            if should_delete:
                deletions.append(DeletionRange(
                    start=curr.start,
                    end=curr.end,
                    reason=reason
                ))

        return deletions

    def _remove_segments(
        self,
        input_path: str,
        output_path: str,
        deletions: list[DeletionRange]
    ) -> str:
        """Remove marked segments from audio."""
        audio = AudioSegment.from_wav(input_path)

        # Sort by start time
        sorted_dels = sorted(deletions, key=lambda x: x.start)

        # Build output by keeping non-deleted parts
        result = AudioSegment.empty()
        current_pos = 0.0

        for d in sorted_dels:
            # Keep audio before this deletion
            if d.start > current_pos:
                start_ms = int(current_pos * 1000)
                end_ms = int(d.start * 1000)
                result += audio[start_ms:end_ms]
            current_pos = max(current_pos, d.end)

        # Keep audio after last deletion
        if current_pos * 1000 < len(audio):
            result += audio[int(current_pos * 1000):]

        result.export(output_path, format="wav")
        return output_path

    def _trim_silences(
        self,
        input_path: str,
        output_path: str,
        threshold_db: float = -40.0,
        min_silence_ms: int = 800,
        target_silence_ms: int = 100
    ) -> str:
        """
        Detect silences > min_silence_ms and truncate to target_silence_ms.

        This keeps pacing natural instead of completely removing silences.
        """
        sample_rate, audio_data = wavfile.read(input_path)

        # Convert to float
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_float = audio_data.astype(np.float32) / 2147483648.0
        else:
            audio_float = audio_data.astype(np.float32)

        if len(audio_float.shape) > 1:
            audio_float = np.mean(audio_float, axis=1)

        # Detect silence regions
        threshold_linear = 10 ** (threshold_db / 20)
        window_size = int(sample_rate * 0.02)  # 20ms windows
        hop_size = window_size // 2

        silence_regions = []
        in_silence = False
        silence_start = 0

        for i in range(0, len(audio_float) - window_size, hop_size):
            window = audio_float[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            current_time = i / sample_rate

            if rms < threshold_linear:
                if not in_silence:
                    in_silence = True
                    silence_start = current_time
            else:
                if in_silence:
                    duration_ms = (current_time - silence_start) * 1000
                    if duration_ms >= min_silence_ms:
                        silence_regions.append((silence_start, current_time))
                    in_silence = False

        # Handle trailing silence
        if in_silence:
            end_time = len(audio_float) / sample_rate
            duration_ms = (end_time - silence_start) * 1000
            if duration_ms >= min_silence_ms:
                silence_regions.append((silence_start, end_time))

        if not silence_regions:
            # No long silences, just copy
            audio = AudioSegment.from_wav(input_path)
            audio.export(output_path, format="wav")
            return output_path

        # Trim silences using pydub
        audio = AudioSegment.from_wav(input_path)
        target_sec = target_silence_ms / 1000.0
        result = audio

        # Process from end to start to avoid position shifts
        for start, end in reversed(silence_regions):
            silence_duration = end - start
            trim_amount = silence_duration - target_sec

            if trim_amount > 0:
                # Keep half of target at start, half at end
                keep_start = target_sec / 2
                keep_end = target_sec / 2

                trim_start_ms = int((start + keep_start) * 1000)
                trim_end_ms = int((end - keep_end) * 1000)

                result = result[:trim_start_ms] + result[trim_end_ms:]

        result.export(output_path, format="wav")
        return output_path
