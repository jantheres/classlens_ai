# 🎓 ClassLens AI — Online Class Quality Evaluator

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.53-red.svg)](https://streamlit.io/)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-green.svg)](https://github.com/openai/whisper)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.13-orange.svg)](https://opencv.org/)
[![Tests](https://img.shields.io/badge/Pytest-Passing-brightgreen.svg)]()

> Built for the **HUMANITI AI Recruitment Build Challenge**.
> Evaluates the pedagogical quality of online classes for children and classifies them into **Excellent**, **Good**, **Average**, or **Poor** using concrete, timestamped evidence.

---

## 🌟 Core Highlights

1. **100% Privacy-Preserving & Local Processing**:
   - Classroom recordings involve children. **Zero raw video, audio, or student names are sent to third-party cloud APIs**.
   - Speech-to-text uses local, offline **OpenAI Whisper** running directly on CPU.
   - Frame sampling and lighting metrics are computed locally via **OpenCV**.
2. **Transparent, Evidence-Based Rubric**:
   - Replaces vague LLM hallucination with a deterministic, weighted scoring engine.
   - Covers 5 core dimensions:
     1. **Teaching Structure & Lesson Flow** (25%)
     2. **Student Interaction & Questioning** (25%)
     3. **Rapport & Positive Reinforcement** (20%)
     4. **Audio Quality & Speech Pacing** (15%)
     5. **Visual Presentation & Presence** (15%)
3. **Missing Evidence vs. Negative Evidence**:
   - Distinguishes between when evidence is *missing* (e.g., absent audio track → reports low confidence and limitations) versus when evidence is *negative* (e.g., audio is loud and clear, but teacher monologues without asking a single question or pausing).
4. **Child Safety Compliance**:
   - Strictly conforms to the challenge rule: **Does NOT infer emotions, student engagement, or learning outcomes from facial expressions.**
   - Only measures objective environmental parameters: teacher visibility %, lighting exposure/contrast, and camera stability.

---

## 📂 Project Architecture & File Guide

```text
ClassLens AI/
├── app.py                # Modern Streamlit user interface
├── analyzer.py           # Video validation, audio extraction & Whisper STT, OpenCV visual metrics
├── scoring.py            # Transparent evidence-based pedagogical rubric & scoring engine
├── test_evaluator.py     # Automated unit & integration tests (Pytest)
├── requirements.txt      # Dependency manifest
├── .env.example          # Safe configuration settings template
├── .gitignore            # Protects child recordings, temporary media, and secrets from Git
└── README.md             # Complete documentation and interview walkthrough
```

### What Each File Does (Beginner-Friendly Explanation)

* **`app.py`**: The visual dashboard the user interacts with in their web browser. It provides video file upload, a progress timeline, overall rating pill badges (`Excellent`, `Good`, `Average`, `Poor`), dimension score cards, timestamped dialogue transcript excerpts, visual frame metrics, and a one-click report export.
* **`analyzer.py`**: The media processing engine. It checks file formats (MP4, MKV, WebM, etc.), uses bundled `imageio-ffmpeg` to extract a clean 16kHz audio track, runs offline Whisper transcription with segment timestamps, and uses OpenCV to assess room lighting and instructor presence.
* **`scoring.py`**: The educational scoring brain. It checks for lesson milestones (greetings, clear objective statements, instructional explanations, comprehension checks, concluding recaps, and homework), calculates words-per-minute pacing, counts teacher questions, and assigns the final weighted quality rating.
* **`test_evaluator.py`**: The test suite that automatically verifies that video validation catches corrupt/empty files, that excellent classes receive high scores, that monologue classes are penalized, and that missing audio is properly flagged.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10 or 3.11 installed.

### 2. Installation
Clone or navigate to the repository folder:
```bash
cd "f:\Downloads\ClassLens AI"
```

Install dependencies:
```bash
pip install -r requirements.txt
```

*(Note: PyAV and imageio-ffmpeg provide all required audio/video decoding binaries automatically. No separate manual installation of system FFmpeg is required!)*

### 3. Run the Automated Tests
Verify all components are working correctly:
```bash
python -m pytest -v
```
You should see:
```text
test_evaluator.py::test_validation_non_existent_file PASSED
test_evaluator.py::test_validation_unsupported_format PASSED
test_evaluator.py::test_validation_empty_file PASSED
test_evaluator.py::test_excellent_class_scoring PASSED
test_evaluator.py::test_poor_monologue_scoring PASSED
test_evaluator.py::test_missing_audio_handling PASSED
test_evaluator.py::test_custom_rubric_weights PASSED
test_evaluator.py::test_synthetic_video_frame_analysis PASSED
8 passed in 0.95s
```

### 4. Launch the Web Application
Start the Streamlit dashboard:
```bash
python -m streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 📊 Pedagogical Quality Rubric

| Category | Score Range | Description |
| :--- | :---: | :--- |
| **Excellent** | **85 – 100** | Exemplary lesson structure (clear objectives, homework, recap), frequent open questioning, positive verbal encouragement, optimal pacing (110–150 WPM), and stable teacher presence with balanced lighting. |
| **Good** | **70 – 84** | Strong instructional delivery and clear explanations with good audiovisuals; minor gaps in concluding summary or interactive pauses. |
| **Average** | **50 – 69** | Acceptable content delivery, but impacted by a predominantly one-way lecture style, infrequent learner questioning, or suboptimal room lighting. |
| **Poor** | **< 50** | Severe pedagogical or technical issues: missing or unintelligible audio, zero interaction, teacher camera completely disabled/obstructed, or abrupt termination without structure. |

---

## 🎯 Technical Interview Talking Points

When presenting this project in a technical interview:

1. **Why Local Whisper + OpenCV instead of Cloud APIs?**
   - *Answer*: When building for children's educational platforms, compliance with child safety laws (e.g., COPPA in the US, GDPR-K in Europe) is paramount. Uploading children's voices or video feeds to third-party endpoints is a major liability. Running lightweight models locally gives total data sovereignty and zero ongoing API costs.
2. **How does the system distinguish Missing Evidence from Negative Evidence?**
   - *Answer*: If a recording has no audio track, that is *missing evidence* — we cannot infer what the teacher said, so the system explicitly lowers the overall confidence score and highlights the limitation. In contrast, if the audio is clear and the teacher speaks for 15 minutes without asking a single question or pausing, that is *negative evidence* — we have positive proof of a non-interactive monologue.
3. **Why avoid facial emotion recognition?**
   - *Answer*: Psychological research indicates that facial geometry does not reliably reflect internal learning states or student engagement. Mandated by ethical AI principles, our visual pipeline only analyzes environmental hygiene: Is the teacher framed and visible? Is the lighting adequate for children to see facial expressions? Is the camera stable?
4. **How are the rubric weights configured?**
   - *Answer*: The weights are exposed in the Streamlit sidebar (Structure 25%, Interaction 25%, Rapport 20%, Audio 15%, Visual 15%). School administrators can adjust the priorities (e.g., prioritizing Interaction for 1-on-1 tutoring, or Structure for larger lecture groups).

---

## 🛡️ License
Built for evaluation under the HUMANITI AI build challenge.
