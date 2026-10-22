"""
ClassLens AI — Media Analyzer Module
====================================
Handles local, privacy-preserving extraction and analysis of video, audio, and speech:
1. Video file validation (format, size, duration, frame integrity).
2. Audio extraction via imageio-ffmpeg / PyAV (16kHz mono WAV).
3. Speech-to-text using local OpenAI Whisper (offline CPU inference).
4. Visual analysis using OpenCV (lighting metrics, face/teacher presence, camera stability).
   NOTE: Compliant with challenge rules — NO emotion or engagement inference from facial expressions.
"""

import os
import subprocess
import tempfile
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

# Try importing imageio_ffmpeg and ensure binary folder is in PATH
try:
    import imageio_ffmpeg
    import shutil
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
    bin_dir = os.path.dirname(FFMPEG_BIN)
    # Ensure ffmpeg.exe exists in the bin_dir for external subprocess calls (e.g. Whisper)
    ffmpeg_alias = os.path.join(bin_dir, "ffmpeg.exe")
    if not os.path.exists(ffmpeg_alias):
        try:
            shutil.copy(FFMPEG_BIN, ffmpeg_alias)
        except Exception:
            pass
    if bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    FFMPEG_BIN = "ffmpeg"


# Global lazy-loaded Whisper model
_WHISPER_MODELS: Dict[str, Any] = {}


def get_whisper_model(model_size: str = "base"):
    """
    Lazy-loads and caches the local Whisper model in memory.
    Runs locally on CPU for complete data privacy.
    """
    global _WHISPER_MODELS
    if model_size not in _WHISPER_MODELS:
        import whisper
        _WHISPER_MODELS[model_size] = whisper.load_model(model_size, device="cpu")
    return _WHISPER_MODELS[model_size]


class VideoValidationError(Exception):
    """Raised when an uploaded video file fails validation checks."""
    pass


def validate_video_file(
    file_path: str,
    max_size_mb: float = 500.0,
    min_duration_sec: float = 5.0
) -> Dict[str, Any]:
    """
    Validates file existence, format extension, size, and basic video stream integrity.
    
    Returns:
        dict: Metadata with duration, fps, resolution, frame_count, file_size_mb.
    Raises:
        VideoValidationError: If file is missing, empty, too large, corrupt, or too short.
    """
    if not os.path.exists(file_path):
        raise VideoValidationError(f"File not found: {file_path}")

    # Check file size
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    if file_size_bytes == 0:
        raise VideoValidationError("The uploaded file is empty (0 bytes).")

    if file_size_mb > max_size_mb:
        raise VideoValidationError(
            f"File size ({file_size_mb:.1f} MB) exceeds maximum allowed limit ({max_size_mb:.0f} MB)."
        )

    # Check extension
    valid_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in valid_extensions:
        raise VideoValidationError(
            f"Unsupported format '{ext}'. Supported formats: {', '.join(sorted(valid_extensions))}"
        )

    # Open with OpenCV to check video stream integrity
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise VideoValidationError("Unable to open video stream. The file may be corrupted or use an unsupported codec.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    duration_sec = 0.0
    if fps > 0 and frame_count > 0:
        duration_sec = frame_count / fps

    cap.release()

    if frame_count <= 0 or width <= 0 or height <= 0:
        raise VideoValidationError("Video has invalid dimensions or zero readable frames.")

    if duration_sec < min_duration_sec:
        raise VideoValidationError(
            f"Video duration ({duration_sec:.1f}s) is too short for a meaningful class evaluation. Minimum is {min_duration_sec:.0f}s."
        )

    return {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "file_size_mb": round(file_size_mb, 2),
        "duration_sec": round(duration_sec, 2),
        "frame_count": frame_count,
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
    }


def parse_vtt_transcript(vtt_file_or_content: str) -> Dict[str, Any]:
    """
    Parses a WebVTT or SRT transcript into structured segments with timestamps and speakers.
    Allows evaluating classes with official challenge ground-truth transcripts.
    """
    import re
    if os.path.exists(vtt_file_or_content):
        with open(vtt_file_or_content, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    else:
        content = vtt_file_or_content

    # Regex matching timestamps 00:01:41.190 --> 00:01:42.590 with optional Speaker: Text
    pattern = re.compile(
        r'(\d{1,2}:\d{2}:\d{2}[\.,]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[\.,]\d{3})\s*\n(?:([^\n:]+):\s*)?(.*?)(?=\n\s*\n|\Z)',
        re.DOTALL
    )

    def to_sec(ts_str: str) -> float:
        ts_clean = ts_str.replace(',', '.')
        parts = ts_clean.split(':')
        if len(parts) == 3:
            return round(int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2]), 2)
        elif len(parts) == 2:
            return round(int(parts[0]) * 60 + float(parts[1]), 2)
        return 0.0

    segments = []
    full_text_list = []
    for m in pattern.finditer(content):
        s_time, e_time, speaker, text = m.groups()
        clean_text = ' '.join(text.split()).strip()
        if clean_text:
            spk = speaker.strip() if speaker else "Speaker"
            formatted_text = f"{spk}: {clean_text}" if speaker else clean_text
            segments.append({
                "start": to_sec(s_time),
                "end": to_sec(e_time),
                "speaker": spk,
                "text": formatted_text,
                "raw_text": clean_text
            })
            full_text_list.append(formatted_text)

    full_text = " ".join(full_text_list)
    words = full_text.split()
    total_words = len(words)
    total_dur = segments[-1]["end"] if segments else 0.0
    wpm = round((total_words / (total_dur / 60.0)), 1) if total_dur > 10 else 0.0

    return {
        "has_speech": len(segments) > 0,
        "transcript_text": full_text,
        "language": "en",
        "segments": segments,
        "word_count": total_words,
        "duration": total_dur,
        "speech_rate_wpm": wpm,
        "volume_stats": {"silence_ratio": 0.25},
        "status_message": f"Loaded {len(segments)} segments from official WebVTT dataset."
    }


def extract_audio(video_path: str, output_wav_path: Optional[str] = None) -> Tuple[bool, Optional[str], str]:
    """
    Extracts audio track from video to 16kHz mono WAV using imageio-ffmpeg.
    
    Returns:
        (has_audio: bool, wav_path: Optional[str], message: str)
    """
    if output_wav_path is None:
        fd, output_wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    # Build ffmpeg command: extract audio to 16kHz, mono, PCM 16-bit
    cmd = [
        FFMPEG_BIN,
        "-y",               # Overwrite output
        "-i", video_path,   # Input file
        "-vn",              # Disable video
        "-acodec", "pcm_s16le",
        "-ar", "16000",     # 16kHz sample rate optimal for Whisper
        "-ac", "1",         # Mono channel
        output_wav_path
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180
        )
        if result.returncode != 0:
            # Check if reason was missing audio stream
            if "does not contain any stream" in result.stderr or "Output file does not contain any stream" in result.stderr:
                if os.path.exists(output_wav_path):
                    try:
                        os.remove(output_wav_path)
                    except OSError:
                        pass
                return False, None, "No audio stream found in this video recording."
            return False, None, f"Audio extraction failed: {result.stderr[:200]}"

        # Check if output file was created and has non-zero size
        if os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 1000:
            return True, output_wav_path, "Audio extracted successfully."
        else:
            return False, None, "Extracted audio file is empty or missing."

    except subprocess.TimeoutExpired:
        return False, None, "Audio extraction timed out after 180 seconds."
    except Exception as e:
        return False, None, f"Audio extraction error: {str(e)}"


def analyze_audio_volume(wav_path: str) -> Dict[str, Any]:
    """
    Performs basic acoustic analysis to detect silence, dead air, and mean volume.
    """
    import wave
    try:
        with wave.open(wav_path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            audio_bytes = wf.readframes(n_frames)

        duration = n_frames / float(framerate) if framerate > 0 else 0.0
        
        # Read raw samples
        if sample_width == 2:
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        else:
            samples = np.frombuffer(audio_bytes, dtype=np.int8).astype(np.float32)

        if len(samples) == 0:
            return {"is_silent": True, "rms_energy": 0.0, "duration": duration, "silence_ratio": 1.0}

        # Calculate RMS energy
        rms = float(np.sqrt(np.mean(samples**2)))
        peak = float(np.max(np.abs(samples)))

        # Frame-wise silence detection (100ms frames)
        frame_len = int(framerate * 0.1)
        if frame_len > 0 and len(samples) >= frame_len:
            num_chunks = len(samples) // frame_len
            chunks = samples[:num_chunks * frame_len].reshape(num_chunks, frame_len)
            chunk_rms = np.sqrt(np.mean(chunks**2, axis=1))
            silence_threshold = 200.0  # Threshold for 16-bit audio
            silent_chunks = np.sum(chunk_rms < silence_threshold)
            silence_ratio = float(silent_chunks / num_chunks)
        else:
            silence_ratio = 0.0

        is_silent = (rms < 150.0) or (silence_ratio > 0.95)

        return {
            "duration": round(duration, 2),
            "rms_energy": round(rms, 1),
            "peak_energy": round(peak, 1),
            "silence_ratio": round(silence_ratio, 2),
            "is_silent": is_silent,
            "framerate": framerate
        }
    except Exception as e:
        return {"is_silent": False, "rms_energy": 0.0, "duration": 0.0, "silence_ratio": 0.0, "error": str(e)}


def transcribe_audio(
    wav_path: Optional[str],
    model_size: str = "base",
    progress_callback=None
) -> Dict[str, Any]:
    """
    Transcribes the extracted audio locally using OpenAI Whisper.
    
    Returns:
        dict: {
            "has_speech": bool,
            "transcript_text": str,
            "language": str,
            "segments": list of {start, end, text, no_speech_prob},
            "word_count": int,
            "duration": float,
            "speech_rate_wpm": float,
            "status_message": str
        }
    """
    if not wav_path or not os.path.exists(wav_path):
        return {
            "has_speech": False,
            "transcript_text": "",
            "language": "unknown",
            "segments": [],
            "word_count": 0,
            "duration": 0.0,
            "speech_rate_wpm": 0.0,
            "status_message": "Audio track missing or not extracted."
        }

    # Acoustic check first
    vol_stats = analyze_audio_volume(wav_path)
    if vol_stats.get("is_silent", False):
        return {
            "has_speech": False,
            "transcript_text": "",
            "language": "unknown",
            "segments": [],
            "word_count": 0,
            "duration": vol_stats.get("duration", 0.0),
            "speech_rate_wpm": 0.0,
            "status_message": "Audio track is completely silent or muted."
        }

    if progress_callback:
        progress_callback(0.2, f"Loading local Whisper ({model_size}) model...")

    model = get_whisper_model(model_size)

    if progress_callback:
        progress_callback(0.4, "Transcribing speech and extracting timestamps...")

    try:
        # Load audio directly into 16kHz float32 numpy array to avoid external subprocess ffmpeg calls
        import wave
        with wave.open(wav_path, 'rb') as wf:
            framerate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)

        if sampwidth == 2:
            audio_array = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 1:
            audio_array = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            audio_array = np.frombuffer(raw_bytes, dtype=np.float32)

        # Run Whisper transcription locally
        result = model.transcribe(
            audio_array,
            verbose=False,
            fp16=False,            # CPU safe
            task="transcribe"
        )

        raw_segments = result.get("segments", [])
        clean_segments = []
        full_text_list = []

        for seg in raw_segments:
            seg_text = seg.get("text", "").strip()
            if seg_text:
                clean_segments.append({
                    "id": seg.get("id"),
                    "start": round(seg.get("start", 0.0), 2),
                    "end": round(seg.get("end", 0.0), 2),
                    "text": seg_text,
                    "no_speech_prob": round(seg.get("no_speech_prob", 0.0), 3)
                })
                full_text_list.append(seg_text)

        full_text = " ".join(full_text_list)
        words = full_text.split()
        word_count = len(words)
        duration = vol_stats.get("duration", 0.0)

        # Words per minute
        wpm = round((word_count / (duration / 60.0)), 1) if duration > 10 else 0.0

        has_speech = word_count >= 3

        return {
            "has_speech": has_speech,
            "transcript_text": full_text,
            "language": result.get("language", "en"),
            "segments": clean_segments,
            "word_count": word_count,
            "duration": duration,
            "speech_rate_wpm": wpm,
            "volume_stats": vol_stats,
            "status_message": "Speech transcribed successfully." if has_speech else "No intelligible speech detected in audio."
        }

    except Exception as e:
        return {
            "has_speech": False,
            "transcript_text": "",
            "language": "error",
            "segments": [],
            "word_count": 0,
            "duration": vol_stats.get("duration", 0.0),
            "speech_rate_wpm": 0.0,
            "volume_stats": vol_stats,
            "status_message": f"Whisper transcription failed: {str(e)}"
        }


def sample_and_analyze_visuals(
    video_path: str,
    target_samples: int = 40,
    max_duration_sec: Optional[float] = None,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Samples frames across the video and performs objective visual quality assessment:
    1. Lighting adequacy (luminance and contrast).
    2. Teacher presence/visibility (using OpenCV Haar Cascade face detector).
    3. Background/framing stability.
    
    IMPORTANT: Adheres to challenge rules — does NOT infer emotions or student
    engagement from facial expressions. Only detects visibility, lighting, and stability.
    
    Returns:
        dict: Summary statistics and timestamped frame observations.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "teacher_visibility_pct": 0.0,
            "adequate_lighting_pct": 0.0,
            "camera_stable_pct": 0.0,
            "avg_brightness": 0.0,
            "avg_contrast": 0.0,
            "total_sampled": 0,
            "observations": [],
            "status_message": "Could not read video frames for visual analysis."
        }

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / fps if fps > 0 else 0.0

    if total_frames <= 0:
        cap.release()
        return {
            "teacher_visibility_pct": 0.0,
            "adequate_lighting_pct": 0.0,
            "camera_stable_pct": 0.0,
            "avg_brightness": 0.0,
            "avg_contrast": 0.0,
            "total_sampled": 0,
            "observations": [],
            "status_message": "Zero video frames found."
        }

    # Load OpenCV face detector for presence detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Determine frame step
    if max_duration_sec and max_duration_sec > 0 and duration_sec > max_duration_sec:
        max_frame = int(min(total_frames - 1, max_duration_sec * fps))
        duration_sec = max_duration_sec
    else:
        max_frame = total_frames - 1

    sample_count = min(target_samples, max_frame + 1)
    frame_indices = np.linspace(0, max_frame, sample_count, dtype=int)

    sampled_observations = []
    face_detected_count = 0
    adequate_light_count = 0
    stable_frame_count = 0
    brightness_list = []
    contrast_list = []

    prev_gray_small = None

    for i, f_idx in enumerate(frame_indices):
        if progress_callback and (i % 5 == 0):
            progress_callback(0.5 + (0.4 * (i / len(frame_indices))), f"Analyzing video frames ({i+1}/{len(frame_indices)})...")

        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        timestamp = round(f_idx / fps, 1)

        # Downscale for fast, robust processing
        h, w = frame.shape[:2]
        target_w = 480
        target_h = int(h * (target_w / w))
        frame_resized = cv2.resize(frame, (target_w, target_h))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # 1. Lighting analysis (mean luminance & standard deviation contrast)
        mean_brightness = float(np.mean(gray))
        contrast_rms = float(np.std(gray))
        brightness_list.append(mean_brightness)
        contrast_list.append(contrast_rms)

        # Lighting classification
        if mean_brightness < 60:
            light_quality = "Poor (Under-exposed / Too Dark)"
            is_light_adequate = False
        elif mean_brightness > 220:
            light_quality = "Poor (Over-exposed / Glare)"
            is_light_adequate = False
        elif contrast_rms < 25:
            light_quality = "Low Contrast (Flat / Washed-out)"
            is_light_adequate = False
        else:
            light_quality = "Adequate"
            is_light_adequate = True

        if is_light_adequate:
            adequate_light_count += 1

        # 2. Teacher presence detection (face bounding box detection)
        # Search primarily in upper 75% where webcam instructor sits
        upper_gray = gray[:int(target_h * 0.85), :]
        faces = face_cascade.detectMultiScale(
            upper_gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30)
        )
        has_face = len(faces) > 0
        if has_face:
            face_detected_count += 1

        # 3. Camera stability / background movement
        gray_small = cv2.resize(gray, (160, 120))
        is_stable = True
        if prev_gray_small is not None:
            diff = cv2.absdiff(gray_small, prev_gray_small)
            motion_score = float(np.mean(diff))
            # If motion score between sampled intervals is moderate/low, camera is stable
            if motion_score > 45.0:
                is_stable = False
        prev_gray_small = gray_small

        if is_stable:
            stable_frame_count += 1

        sampled_observations.append({
            "timestamp": timestamp,
            "frame_index": int(f_idx),
            "brightness": round(mean_brightness, 1),
            "contrast": round(contrast_rms, 1),
            "lighting_status": light_quality,
            "is_lighting_adequate": is_light_adequate,
            "teacher_present": has_face,
            "camera_stable": is_stable
        })

    cap.release()

    total_valid = len(sampled_observations)
    if total_valid == 0:
        return {
            "teacher_visibility_pct": 0.0,
            "adequate_lighting_pct": 0.0,
            "camera_stable_pct": 0.0,
            "avg_brightness": 0.0,
            "avg_contrast": 0.0,
            "total_sampled": 0,
            "observations": [],
            "status_message": "Could not successfully decode sampled video frames."
        }

    teacher_vis_pct = round((face_detected_count / total_valid) * 100, 1)
    adequate_light_pct = round((adequate_light_count / total_valid) * 100, 1)
    camera_stable_pct = round((stable_frame_count / total_valid) * 100, 1)
    avg_brightness = round(float(np.mean(brightness_list)), 1)
    avg_contrast = round(float(np.mean(contrast_list)), 1)

    return {
        "teacher_visibility_pct": teacher_vis_pct,
        "adequate_lighting_pct": adequate_light_pct,
        "camera_stable_pct": camera_stable_pct,
        "avg_brightness": avg_brightness,
        "avg_contrast": avg_contrast,
        "total_sampled": total_valid,
        "observations": sampled_observations,
        "status_message": f"Analyzed {total_valid} video frames across {round(duration_sec, 1)} seconds."
    }


def analyze_recording(
    video_path: str,
    whisper_model_size: str = "base",
    progress_callback=None
) -> Dict[str, Any]:
    """
    Main orchestration pipeline:
    1. Validates video file.
    2. Extracts audio locally.
    3. Transcribes speech locally via Whisper.
    4. Analyzes visual frames locally via OpenCV.
    5. Safely cleans up temporary audio artifacts.
    """
    if progress_callback:
        progress_callback(0.05, "Validating video file...")

    video_meta = validate_video_file(video_path)

    if progress_callback:
        progress_callback(0.15, "Extracting audio track locally...")

    has_audio, wav_path, audio_msg = extract_audio(video_path)

    audio_res = {}
    try:
        if has_audio and wav_path:
            audio_res = transcribe_audio(
                wav_path,
                model_size=whisper_model_size,
                progress_callback=progress_callback
            )
        else:
            audio_res = {
                "has_speech": False,
                "transcript_text": "",
                "language": "none",
                "segments": [],
                "word_count": 0,
                "duration": video_meta["duration_sec"],
                "speech_rate_wpm": 0.0,
                "status_message": audio_msg
            }
    finally:
        # Safely remove extracted WAV to preserve disk and child privacy
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass

    if progress_callback:
        progress_callback(0.55, "Sampling and analyzing video frames...")

    visual_res = sample_and_analyze_visuals(
        video_path,
        target_samples=30,
        progress_callback=progress_callback
    )

    if progress_callback:
        progress_callback(0.95, "Media analysis complete!")

    return {
        "video_metadata": video_meta,
        "audio_analysis": audio_res,
        "visual_analysis": visual_res
    }
