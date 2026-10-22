"""
Unit & Integration Tests for ClassLens AI
=========================================
Tests:
1. Video file validation (invalid extension, non-existent, empty).
2. Evidence-based scoring engine (Excellent, Good, Average, Poor scenarios).
3. Monologue detection (negative evidence vs missing evidence).
4. Missing audio handling (confidence drop and limitations reporting).
5. Synthetic video creation and visual analyzer frame extraction.
"""

import os
import tempfile
import pytest
import numpy as np
import cv2

from analyzer import (
    validate_video_file,
    VideoValidationError,
    sample_and_analyze_visuals,
    extract_audio
)
from scoring import (
    compute_class_evaluation,
    evaluate_structure,
    evaluate_interaction,
    evaluate_rapport,
    evaluate_audio_quality,
    evaluate_visual_presentation,
    DEFAULT_WEIGHTS
)


# -------------------------------------------------------------
# 1. Validation Tests
# -------------------------------------------------------------
def test_validation_non_existent_file():
    with pytest.raises(VideoValidationError, match="File not found"):
        validate_video_file("non_existent_video_path.mp4")


def test_validation_unsupported_format():
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(path, "w") as f:
            f.write("hello")
        with pytest.raises(VideoValidationError, match="Unsupported format"):
            validate_video_file(path)
    finally:
        os.remove(path)


def test_validation_empty_file():
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        with pytest.raises(VideoValidationError, match="empty"):
            validate_video_file(path)
    finally:
        os.remove(path)


# -------------------------------------------------------------
# 2. Scoring Logic Tests
# -------------------------------------------------------------
def test_excellent_class_scoring():
    """Verify an exemplary class with greeting, questions, praise, and homework scores Excellent (>=85)."""
    segments = [
        {"start": 1.0, "end": 4.0, "text": "Good morning everybody, welcome to today's math lesson!"},
        {"start": 5.0, "end": 8.0, "text": "Today we will learn how fractions work step by step."},
        {"start": 12.0, "end": 15.0, "text": "Can you tell me what happens if we split the apple in two?"},
        {"start": 17.0, "end": 20.0, "text": "Excellent! Great job, that is exactly right."},
        {"start": 22.0, "end": 26.0, "text": "Notice that because both sides are equal, each part is one half."},
        {"start": 29.0, "end": 32.0, "text": "Do you understand? Are there any questions so far?"},
        {"start": 35.0, "end": 38.0, "text": "Awesome participation! Well done today everyone."},
        {"start": 40.0, "end": 45.0, "text": "In conclusion, fractions represent parts of a whole."},
        {"start": 46.0, "end": 50.0, "text": "Please complete your homework worksheet before our next class."}
    ]

    analysis = {
        "video_metadata": {"duration_sec": 55.0},
        "audio_analysis": {
            "has_speech": True,
            "word_count": 115,
            "speech_rate_wpm": 125.0,
            "language": "en",
            "segments": segments,
            "volume_stats": {"silence_ratio": 0.25}
        },
        "visual_analysis": {
            "total_sampled": 20,
            "teacher_visibility_pct": 95.0,
            "adequate_lighting_pct": 90.0,
            "camera_stable_pct": 95.0,
            "avg_brightness": 120.0,
            "observations": []
        }
    }

    result = compute_class_evaluation(analysis)
    assert result["overall_score"] >= 85.0
    assert result["rating"] == "Excellent"
    assert len(result["positive_observations"]) > 0
    assert result["confidence"]["level"] == "High"


def test_poor_monologue_scoring():
    """Verify that a monologue with zero questions, zero praise, and zero structure is penalized."""
    # Teacher talks constantly without pausing, asking questions, or praising
    segments = [
        {"start": 0.0, "end": 15.0, "text": "Now look at the page and read the text silently as I turn pages."},
        {"start": 15.1, "end": 30.0, "text": "I will proceed to read the paragraphs one through ten without stop."},
        {"start": 30.1, "end": 45.0, "text": "The text discusses ancient empires and historical dates in chronology."},
        {"start": 45.1, "end": 60.0, "text": "The chapter is finished."}
    ]

    analysis = {
        "video_metadata": {"duration_sec": 60.0},
        "audio_analysis": {
            "has_speech": True,
            "word_count": 80,
            "speech_rate_wpm": 80.0,
            "language": "en",
            "segments": segments,
            "volume_stats": {"silence_ratio": 0.05}
        },
        "visual_analysis": {
            "total_sampled": 20,
            "teacher_visibility_pct": 10.0,
            "adequate_lighting_pct": 40.0,
            "camera_stable_pct": 50.0,
            "avg_brightness": 45.0,
            "observations": []
        }
    }

    result = compute_class_evaluation(analysis)
    assert result["overall_score"] < 50.0
    assert result["rating"] == "Poor"
    # Negative evidence for interaction should be noted in improvement areas
    assert any("NEGATIVE EVIDENCE" in s or "questioning" in s.lower() for s in result["improvement_areas"])


def test_missing_audio_handling():
    """Verify that absent audio track results in missing evidence reporting and dropped confidence."""
    analysis = {
        "video_metadata": {"duration_sec": 45.0},
        "audio_analysis": {
            "has_speech": False,
            "word_count": 0,
            "speech_rate_wpm": 0.0,
            "language": "none",
            "segments": [],
            "status_message": "No audio stream found in this video recording."
        },
        "visual_analysis": {
            "total_sampled": 15,
            "teacher_visibility_pct": 80.0,
            "adequate_lighting_pct": 85.0,
            "camera_stable_pct": 90.0,
            "avg_brightness": 110.0,
            "observations": []
        }
    }

    result = compute_class_evaluation(analysis)
    assert result["confidence"]["level"] == "Low"
    assert any("Missing audio evidence" in lim for lim in result["confidence"]["limitations"])


def test_custom_rubric_weights():
    """Verify configurable weights properly shift score contribution."""
    segments = [
        {"start": 1.0, "end": 5.0, "text": "Hello class, welcome today!"}
    ]
    analysis = {
        "video_metadata": {"duration_sec": 30.0},
        "audio_analysis": {
            "has_speech": True,
            "word_count": 10,
            "speech_rate_wpm": 120.0,
            "language": "en",
            "segments": segments,
            "volume_stats": {"silence_ratio": 0.2}
        },
        "visual_analysis": {
            "total_sampled": 10,
            "teacher_visibility_pct": 100.0,
            "adequate_lighting_pct": 100.0,
            "camera_stable_pct": 100.0,
            "avg_brightness": 120.0,
            "observations": []
        }
    }

    # High visual weight should reward 100% visual metrics
    weights_visual_heavy = {
        "structure": 0.05,
        "interaction": 0.05,
        "rapport": 0.05,
        "audio": 0.05,
        "visual": 0.80
    }
    res_visual = compute_class_evaluation(analysis, custom_weights=weights_visual_heavy)
    res_default = compute_class_evaluation(analysis, custom_weights=DEFAULT_WEIGHTS)

    assert res_visual["overall_score"] > res_default["overall_score"]


# -------------------------------------------------------------
# 3. Synthetic Video Visual Analysis Test
# -------------------------------------------------------------
def test_synthetic_video_frame_analysis():
    """Creates a temporary synthetic video and tests OpenCV sampling and lighting assessment."""
    fd, video_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    try:
        width, height = 320, 240
        fps = 10
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        # Generate 60 frames (6 seconds) with varying brightness
        for i in range(60):
            brightness = int(50 + (i * 2))  # Range 50 to 170
            frame = np.full((height, width, 3), brightness, dtype=np.uint8)
            # Draw a circle simulating a face in upper area
            cv2.circle(frame, (width // 2, height // 3), 40, (220, 220, 220), -1)
            out.write(frame)
        out.release()

        # Test video validation
        meta = validate_video_file(video_path, min_duration_sec=3.0)
        assert meta["duration_sec"] >= 5.0
        assert meta["width"] == 320
        assert meta["height"] == 240

        # Test visual frame analysis
        visual_res = sample_and_analyze_visuals(video_path, target_samples=10)
        assert visual_res["total_sampled"] > 0
        assert visual_res["avg_brightness"] > 0

        # Test audio extraction on video without audio
        has_audio, wav_path, msg = extract_audio(video_path)
        assert has_audio is False  # Synthetic video has no audio track
        assert "No audio stream" in msg or "stream" in msg.lower() or "failed" in msg.lower()

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


# -------------------------------------------------------------
# 4. WebVTT Parser & Challenge Dataset Ground Truth Test
# -------------------------------------------------------------
def test_parse_vtt_transcript():
    """Verifies that parse_vtt_transcript correctly extracts timestamps, speakers, and segments."""
    from analyzer import parse_vtt_transcript

    sample_vtt = """WEBVTT

1
00:01:41.190 --> 00:01:42.590
Ahna: Good evening, sir!

2
00:01:49.040 --> 00:01:50.569
Sumit Ranjan: Hey, good evening, Anna.
"""
    parsed = parse_vtt_transcript(sample_vtt)
    assert parsed["has_speech"] is True
    assert len(parsed["segments"]) == 2
    assert parsed["segments"][0]["speaker"] == "Ahna"
    assert parsed["segments"][0]["start"] == 101.19
    assert parsed["segments"][1]["speaker"] == "Sumit Ranjan"
    assert parsed["word_count"] > 0

