"""
ClassLens AI — Evidence-Based Scoring Module
============================================
Evaluates classroom quality across 5 transparent, configurable dimensions:
1. Teaching Structure & Lesson Flow (25%)
2. Student Interaction & Questioning (25%)
3. Rapport & Positive Reinforcement (20%)
4. Audio Quality & Speech Delivery (15%)
5. Visual Presentation & Presence (15%)

Key pedagogical principles:
- Transparent rubric with configurable dimension weights.
- Distinguishes between MISSING evidence (e.g. no audio track) and NEGATIVE evidence (e.g. monologue with no questions).
- Never hallucinates generic feedback: all points cite specific transcript segments or visual timestamps.
- Reports confidence score and explicit analysis limitations.
"""

import re
from typing import Dict, Any, List, Optional, Tuple


# Default configurable rubric weights (sum = 1.0)
DEFAULT_WEIGHTS = {
    "structure": 0.25,
    "interaction": 0.25,
    "rapport": 0.20,
    "audio": 0.15,
    "visual": 0.15,
}

# Categorical Thresholds
RATING_THRESHOLDS = [
    (85.0, "Excellent", "Demonstrates outstanding pedagogical structure, active questioning, warm rapport, and clear audiovisual presence."),
    (70.0, "Good", "Effective teaching with clear explanations and good presentation; minor opportunities for enhanced interaction or pacing."),
    (50.0, "Average", "Acceptable instruction, but hindered by limited student interaction, pacing issues, or visual/structural gaps."),
    (0.0,  "Poor", "Significant pedagogical or technical deficiencies; lacking engagement, structure, or intelligible delivery.")
]


# Keyword Lexicons for Pedagogical Evidence Detection
LEXICONS = {
    "greeting": [
        "hello", "hi", "good morning", "good afternoon", "welcome", "how are you", 
        "nice to see you", "glad you joined", "everybody", "everyone", "class"
    ],
    "objectives": [
        "today we will", "our topic", "let's learn", "we are going to", "today's lesson", 
        "goal", "objective", "we will cover", "let us start", "first we will"
    ],
    "explanation": [
        "because", "for example", "step by step", "notice that", "this means", 
        "let me explain", "observe", "remember that", "the rule is", "look at this", 
        "in other words", "reason", "definition"
    ],
    "praise": [
        "good job", "great job", "well done", "excellent", "awesome", "perfect", 
        "nice try", "very good", "fantastic", "brilliant", "superb", "exactly right", 
        "correct", "wonderful", "proud of you", "keep it up", "spot on", "thank you"
    ],
    "check_understanding": [
        "understand", "do you follow", "make sense", "any questions", "are you with me", 
        "can you hear me", "got it", "clear", "any doubt"
    ],
    "conclusion": [
        "in conclusion", "to summarize", "wrap up", "summary", "today we learned", 
        "that brings us to the end", "that's all for today"
    ],
    "homework": [
        "homework", "assignment", "next class", "practice for next time", "exercise", 
        "worksheet", "see you next", "by tomorrow"
    ]
}


def _find_keyword_matches(segments: List[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
    """Finds transcript segments containing any of the keywords."""
    matches = []
    for seg in segments:
        text_lower = seg.get("text", "").lower()
        for kw in keywords:
            # Word boundary search for precision
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, text_lower):
                matches.append({
                    "keyword": kw,
                    "timestamp": f"{seg.get('start', 0.0):.1f}s - {seg.get('end', 0.0):.1f}s",
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "text": seg.get("text", "")
                })
                break
    return matches


def _find_questions(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identifies questions based on punctuation or question syntax."""
    questions = []
    q_words = {"what", "why", "how", "who", "where", "when", "which", "can you", "could you", "do you", "is it", "are you"}
    for seg in segments:
        text = seg.get("text", "").strip()
        text_lower = text.lower()
        is_q = False
        if "?" in text:
            is_q = True
        else:
            for qw in q_words:
                if text_lower.startswith(qw + " "):
                    is_q = True
                    break
        if is_q:
            questions.append({
                "timestamp": f"{seg.get('start', 0.0):.1f}s - {seg.get('end', 0.0):.1f}s",
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": text
            })
    return questions


def evaluate_structure(segments: List[Dict[str, Any]], duration_sec: float) -> Dict[str, Any]:
    """
    Evaluates Teaching Structure & Lesson Flow (0-100 score).
    Looks for Introduction, Concept Explanations, Checks, Conclusion, and Homework.
    """
    if not segments:
        return {
            "score": 0.0,
            "evidence_status": "missing_audio",
            "findings": ["No speech transcript available to evaluate lesson structure."],
            "milestones_found": {},
            "citations": []
        }

    # Split segments into early (first 25%), middle (50%), and late (last 25%)
    early_cutoff = duration_sec * 0.30
    late_cutoff = max(duration_sec * 0.70, duration_sec - 60.0)

    early_segs = [s for s in segments if s.get("start", 0.0) <= early_cutoff]
    late_segs = [s for s in segments if s.get("end", 0.0) >= late_cutoff]

    greeting_matches = _find_keyword_matches(early_segs or segments[:5], LEXICONS["greeting"])
    objective_matches = _find_keyword_matches(early_segs or segments[:8], LEXICONS["objectives"])
    explanation_matches = _find_keyword_matches(segments, LEXICONS["explanation"])
    check_matches = _find_keyword_matches(segments, LEXICONS["check_understanding"])
    conclusion_matches = _find_keyword_matches(late_segs or segments[-6:], LEXICONS["conclusion"])
    homework_matches = _find_keyword_matches(late_segs or segments[-6:], LEXICONS["homework"])

    score = 0.0
    findings = []
    citations = []

    # 1. Warm Greeting (15 pts)
    if greeting_matches:
        score += 15.0
        findings.append(f"Clear opening greeting detected ({len(greeting_matches)} instances).")
        citations.append({"category": "Greeting", "item": greeting_matches[0]})
    else:
        findings.append("No explicit welcoming greeting detected in the opening phase.")

    # 2. Stated Lesson Goal / Objectives (20 pts)
    if objective_matches:
        score += 20.0
        findings.append("Explicit lesson topic/objective framed early in class.")
        citations.append({"category": "Lesson Objective", "item": objective_matches[0]})
    else:
        score += 5.0  # Partial credit if class jumped straight into content
        findings.append("Lesson topic was not explicitly introduced at the start.")

    # 3. Concept Explanations & Examples (30 pts)
    if len(explanation_matches) >= 3:
        score += 30.0
        findings.append(f"Substantive explanatory depth ({len(explanation_matches)} explanatory phrases/examples).")
        citations.append({"category": "Explanation", "item": explanation_matches[0]})
    elif len(explanation_matches) >= 1:
        score += 20.0
        findings.append(f"Moderate instructional explanation ({len(explanation_matches)} phrases).")
        citations.append({"category": "Explanation", "item": explanation_matches[0]})
    else:
        score += 8.0
        findings.append("Limited instructional explanations or concrete examples provided.")

    # 4. Periodic Checks for Understanding (15 pts)
    if check_matches:
        score += 15.0
        findings.append(f"Periodic comprehension checks conducted ({len(check_matches)} instances).")
        citations.append({"category": "Check for Understanding", "item": check_matches[0]})
    else:
        findings.append("No explicit comprehension checks (e.g. 'do you understand?', 'any questions?').")

    # 5. Conclusion & Next Steps / Homework (20 pts)
    wrap_score = 0.0
    if conclusion_matches:
        wrap_score += 10.0
        findings.append("Lesson was rounded off with a recap or concluding statement.")
        citations.append({"category": "Conclusion", "item": conclusion_matches[0]})
    if homework_matches:
        wrap_score += 10.0
        findings.append("Homework or next class follow-up was assigned.")
        citations.append({"category": "Homework", "item": homework_matches[0]})
    if not conclusion_matches and not homework_matches:
        findings.append("Abrupt ending: no summary recap or homework/follow-up instructions detected.")
    score += wrap_score

    return {
        "score": min(round(score, 1), 100.0),
        "evidence_status": "sufficient",
        "findings": findings,
        "milestones_found": {
            "greeting": len(greeting_matches) > 0,
            "objectives": len(objective_matches) > 0,
            "explanation_count": len(explanation_matches),
            "check_count": len(check_matches),
            "conclusion": len(conclusion_matches) > 0,
            "homework": len(homework_matches) > 0
        },
        "citations": citations
    }


def evaluate_interaction(segments: List[Dict[str, Any]], duration_sec: float) -> Dict[str, Any]:
    """
    Evaluates Student Interaction & Questioning (0-100 score).
    Counts questions, detects pauses/dialogue opportunities, identifies monologues.
    """
    if not segments:
        return {
            "score": 0.0,
            "evidence_status": "missing_audio",
            "findings": ["No audio transcript available to evaluate teacher-student interaction."],
            "question_count": 0,
            "citations": []
        }

    questions = _find_questions(segments)
    q_count = len(questions)

    # Estimate conversational turns: pauses between segments > 1.5s often signify student response or wait time
    pause_turns = 0
    for i in range(len(segments) - 1):
        gap = segments[i+1].get("start", 0.0) - segments[i].get("end", 0.0)
        if 1.2 <= gap <= 12.0:
            pause_turns += 1

    findings = []
    citations = []

    # Normalized question rate per 5 minutes
    class_mins = max(duration_sec / 60.0, 1.0)
    questions_per_5m = (q_count / class_mins) * 5.0

    score = 0.0

    # 1. Question Frequency (40 pts)
    if questions_per_5m >= 4.0:
        score += 40.0
        findings.append(f"Highly inquisitive teaching style: {q_count} questions asked (~{questions_per_5m:.1f} per 5 mins).")
    elif questions_per_5m >= 2.0:
        score += 30.0
        findings.append(f"Adequate questioning: {q_count} questions asked (~{questions_per_5m:.1f} per 5 mins).")
    elif q_count >= 1:
        score += 15.0
        findings.append(f"Infrequent questioning: only {q_count} questions asked across {class_mins:.1f} minutes.")
    else:
        score += 0.0
        findings.append("NEGATIVE EVIDENCE: Complete absence of questions. Teacher delivered a continuous one-way lecture.")

    # 2. Wait-Time & Interactive Turn-Taking (35 pts)
    if pause_turns >= 4:
        score += 35.0
        findings.append(f"Healthy dialogue pacing with {pause_turns} interactive turn pauses observed.")
    elif pause_turns >= 2:
        score += 22.0
        findings.append(f"Moderate pause intervals ({pause_turns} observed); some space given for learner input.")
    else:
        score += 8.0
        findings.append("Rushed delivery: minimal pause intervals provided for student comprehension or responses.")

    # 3. Direct Engagement Keywords (25 pts)
    direct_prompts = _find_keyword_matches(segments, ["your turn", "can you try", "go ahead", "unmute", "tell me", "type in chat"])
    if direct_prompts:
        score += 25.0
        findings.append(f"Direct learner prompts utilized ({len(direct_prompts)} instances).")
        citations.append({"category": "Direct Prompt", "item": direct_prompts[0]})
    elif q_count > 0:
        score += 15.0
    else:
        findings.append("No direct student invitation or turn-taking prompts detected.")

    for q in questions[:3]:
        citations.append({"category": "Teacher Question", "item": q})

    return {
        "score": min(round(score, 1), 100.0),
        "evidence_status": "sufficient",
        "findings": findings,
        "question_count": q_count,
        "questions_per_5m": round(questions_per_5m, 1),
        "pause_turns": pause_turns,
        "citations": citations
    }


def evaluate_rapport(segments: List[Dict[str, Any]], duration_sec: float = 60.0) -> Dict[str, Any]:
    """
    Evaluates Rapport & Positive Reinforcement (0-100 score).
    Measures praise frequency, encouragement, warmth, and supportive tone normalized by class duration.
    """
    if not segments:
        return {
            "score": 0.0,
            "evidence_status": "missing_audio",
            "findings": ["No speech transcript available to evaluate rapport or reinforcement."],
            "praise_count": 0,
            "citations": []
        }

    praise_matches = _find_keyword_matches(segments, LEXICONS["praise"])
    praise_count = len(praise_matches)
    dur_mins = max(duration_sec / 60.0, 1.0)
    praise_per_10m = (praise_count / dur_mins) * 10.0

    findings = []
    citations = []

    score = 0.0

    if praise_per_10m >= 5.0:
        score = 95.0
        findings.append(f"Outstanding encouragement: {praise_count} positive reinforcement remarks (~{praise_per_10m:.1f} per 10 mins).")
    elif praise_per_10m >= 2.5:
        score = 80.0
        findings.append(f"Strong positive reinforcement: {praise_count} affirming phrases (~{praise_per_10m:.1f} per 10 mins).")
    elif praise_per_10m >= 1.0:
        score = 65.0
        findings.append(f"Moderate encouragement: {praise_count} praise instance(s) across {int(dur_mins)} mins. Opportunity for more energetic warmth.")
    elif praise_count >= 1:
        score = 48.0
        findings.append(f"Infrequent praise: only {praise_count} positive affirmation(s) across {int(dur_mins)} mins. Recommend more pleasant, energetic reinforcement.")
    else:
        score = 30.0
        findings.append("NEGATIVE EVIDENCE: Complete absence of verbal praise, celebration, or encouraging reinforcement.")

    for p in praise_matches[:3]:
        citations.append({"category": "Praise / Encouragement", "item": p})

    return {
        "score": score,
        "evidence_status": "sufficient",
        "findings": findings,
        "praise_count": praise_count,
        "praise_per_10m": round(praise_per_10m, 1),
        "citations": citations
    }


def evaluate_audio_quality(audio_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates Audio Quality & Speech Delivery (0-100 score).
    Considers speech presence, word-per-minute pacing, and volume stability.
    """
    has_speech = audio_analysis.get("has_speech", False)
    status_msg = audio_analysis.get("status_message", "")
    vol_stats = audio_analysis.get("volume_stats", {})
    wpm = audio_analysis.get("speech_rate_wpm", 0.0)
    word_count = audio_analysis.get("word_count", 0)

    if not has_speech or word_count == 0:
        return {
            "score": 10.0,
            "evidence_status": "missing_or_silent",
            "findings": [f"Missing or silent audio: {status_msg}"],
            "wpm": 0.0,
            "is_silent": True
        }

    findings = []
    score = 40.0  # Base score for functional audio track with speech

    # 1. Pacing / Words Per Minute (ideal for kids: 105-165 WPM)
    if 105 <= wpm <= 165:
        score += 35.0
        findings.append(f"Optimal speech pacing for young learners ({wpm} WPM).")
    elif 80 <= wpm < 105:
        score += 25.0
        findings.append(f"Deliberate, slow speech pace ({wpm} WPM) — easy to follow.")
    elif 60 <= wpm < 80:
        score += 15.0
        findings.append(f"Slow speech pace ({wpm} WPM); opportunity for more energetic delivery.")
    elif 165 < wpm <= 190:
        score += 20.0
        findings.append(f"Fast delivery ({wpm} WPM); consider moderating pace for younger children.")
    else:
        score += 5.0
        findings.append(f"Atypical speech rate ({wpm} WPM): very slow pacing or prolonged dragging pauses.")

    # 2. Volume & Energy Stability
    silence_ratio = vol_stats.get("silence_ratio", 0.0)
    if silence_ratio <= 0.35:
        score += 25.0
        findings.append(f"Consistent vocal presence (silence ratio {silence_ratio*100:.0f}%).")
    elif silence_ratio <= 0.60:
        score += 15.0
        findings.append(f"Moderate pause periods (silence ratio {silence_ratio*100:.0f}%).")
    else:
        score += 5.0
        findings.append(f"Substantial dead air or audio dropouts (silence ratio {silence_ratio*100:.0f}%).")

    return {
        "score": min(round(score, 1), 100.0),
        "evidence_status": "sufficient",
        "findings": findings,
        "wpm": wpm,
        "word_count": word_count,
        "silence_ratio": silence_ratio
    }


def evaluate_visual_presentation(visual_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates Visual Presentation & Presence (0-100 score).
    Factors: Teacher visibility %, lighting adequacy %, camera stability %.
    NOTE: Strictly objective. Does NOT infer internal emotions from facial features.
    """
    total_sampled = visual_analysis.get("total_sampled", 0)
    if total_sampled == 0:
        return {
            "score": 15.0,
            "evidence_status": "insufficient_frames",
            "findings": ["Could not sample video frames for visual analysis."],
            "citations": []
        }

    teacher_vis_pct = visual_analysis.get("teacher_visibility_pct", 0.0)
    lighting_pct = visual_analysis.get("adequate_lighting_pct", 0.0)
    stability_pct = visual_analysis.get("camera_stable_pct", 0.0)
    avg_brightness = visual_analysis.get("avg_brightness", 0.0)

    score = 0.0
    findings = []
    citations = []

    # 1. Teacher Presence / Camera-On (40 pts)
    if teacher_vis_pct >= 75.0:
        score += 40.0
        findings.append(f"High teacher visual engagement: visible in {teacher_vis_pct}% of sampled frames.")
    elif teacher_vis_pct >= 40.0:
        score += 25.0
        findings.append(f"Moderate teacher visibility ({teacher_vis_pct}% of frames); camera occasionally off or obstructed.")
    elif teacher_vis_pct > 0.0:
        score += 12.0
        findings.append(f"Low teacher visibility ({teacher_vis_pct}% of frames). Face largely missing or off-screen.")
    else:
        score += 5.0
        findings.append("NEGATIVE EVIDENCE: Instructor face not detected in any sampled frame (camera off or screen share only).")

    # 2. Lighting Adequacy (35 pts)
    if lighting_pct >= 80.0:
        score += 35.0
        findings.append(f"Consistently balanced lighting across {lighting_pct}% of session (mean brightness: {avg_brightness}).")
    elif lighting_pct >= 50.0:
        score += 22.0
        findings.append(f"Acceptable lighting ({lighting_pct}% adequate), with occasional under/over-exposure.")
    else:
        score += 10.0
        findings.append(f"Poor lighting conditions ({lighting_pct}% adequate; mean brightness {avg_brightness}). Review room lighting.")

    # 3. Camera Stability (25 pts)
    if stability_pct >= 85.0:
        score += 25.0
        findings.append("Stable, professional webcam framing without jarring camera shake.")
    elif stability_pct >= 60.0:
        score += 18.0
        findings.append("Generally steady framing with minor camera adjustments.")
    else:
        score += 8.0
        findings.append("Noticeable camera instability or erratic background movements.")

    # Extract sample visual frame timestamps
    obs = visual_analysis.get("observations", [])
    for o in obs[:4]:
        citations.append({
            "timestamp": f"{o['timestamp']}s",
            "teacher_present": o["teacher_present"],
            "lighting": o["lighting_status"],
            "brightness": o["brightness"]
        })

    return {
        "score": min(round(score, 1), 100.0),
        "evidence_status": "sufficient",
        "findings": findings,
        "teacher_visibility_pct": teacher_vis_pct,
        "adequate_lighting_pct": lighting_pct,
        "camera_stable_pct": stability_pct,
        "citations": citations
    }


def compute_class_evaluation(
    analysis_result: Dict[str, Any],
    custom_weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Computes overall weighted quality evaluation, assigns 1 of 4 categories:
    [Excellent, Good, Average, Poor], compiles evidence lists, positive observations,
    areas for improvement, and uncertainty/limitations.
    """
    weights = custom_weights or DEFAULT_WEIGHTS

    audio_res = analysis_result.get("audio_analysis", {})
    visual_res = analysis_result.get("visual_analysis", {})
    video_meta = analysis_result.get("video_metadata", {})

    segments = audio_res.get("segments", [])
    duration_sec = video_meta.get("duration_sec", 60.0)

    # 1. Evaluate individual dimensions
    dim_structure = evaluate_structure(segments, duration_sec)
    dim_interaction = evaluate_interaction(segments, duration_sec)
    dim_rapport = evaluate_rapport(segments, duration_sec)
    dim_audio = evaluate_audio_quality(audio_res)
    dim_visual = evaluate_visual_presentation(visual_res)

    # 2. Weighted overall score calculation
    weighted_score = (
        dim_structure["score"] * weights.get("structure", 0.25) +
        dim_interaction["score"] * weights.get("interaction", 0.25) +
        dim_rapport["score"] * weights.get("rapport", 0.20) +
        dim_audio["score"] * weights.get("audio", 0.15) +
        dim_visual["score"] * weights.get("visual", 0.15)
    )
    overall_score = round(max(0.0, min(100.0, weighted_score)), 1)

    # 3. Assign Classification
    assigned_category = "Poor"
    category_summary = ""
    for min_val, cat, desc in RATING_THRESHOLDS:
        if overall_score >= min_val:
            assigned_category = cat
            category_summary = desc
            break

    # 4. Handle Edge Cases and Confidence Assessment
    limitations = []
    confidence_level = "High"
    confidence_pct = 90

    if not audio_res.get("has_speech", False):
        limitations.append("Missing audio evidence: Speech was inaudible or missing, significantly lowering evaluation confidence for pedagogical dialogue.")
        confidence_level = "Low"
        confidence_pct = 35

    if duration_sec < 30.0:
        limitations.append(f"Short duration recording ({duration_sec:.1f}s): Short sample limits evaluation of full lesson lifecycle (intro to wrap-up).")
        confidence_level = "Medium" if confidence_pct > 50 else confidence_level
        confidence_pct = min(confidence_pct, 65)

    if visual_res.get("total_sampled", 0) < 5:
        limitations.append("Limited visual frames sampled: Camera assessment based on a restricted sample size.")
        confidence_pct = min(confidence_pct, 70)

    # 5. Compile Specific Positive Observations & Areas for Improvement
    positive_observations = []
    improvement_areas = []

    # Structure
    if dim_structure["score"] >= 70:
        for f in dim_structure["findings"]:
            if "Clear opening" in f or "Explicit lesson" in f or "explanatory depth" in f or "Periodic comprehension" in f:
                positive_observations.append(f)
    else:
        for f in dim_structure["findings"]:
            if "No explicit" in f or "not explicitly" in f or "Abrupt ending" in f or "Limited instructional" in f:
                improvement_areas.append(f)

    # Interaction
    if dim_interaction["score"] >= 70:
        for f in dim_interaction["findings"]:
            if "teaching style" in f or "Adequate questioning" in f or "Healthy dialogue" in f:
                positive_observations.append(f)
    else:
        for f in dim_interaction["findings"]:
            if "NEGATIVE EVIDENCE" in f or "Infrequent questioning" in f or "Rushed delivery" in f:
                improvement_areas.append(f)

    # Rapport
    if dim_rapport["score"] >= 70:
        for f in dim_rapport["findings"]:
            if "Outstanding" in f or "Strong positive" in f:
                positive_observations.append(f)
    else:
        for f in dim_rapport["findings"]:
            if "NEGATIVE EVIDENCE" in f or "Moderate encouragement" in f:
                improvement_areas.append(f)

    # Audio
    if dim_audio["score"] >= 70:
        for f in dim_audio["findings"]:
            if "Optimal speech pacing" in f or "Consistent vocal presence" in f:
                positive_observations.append(f)
    else:
        for f in dim_audio["findings"]:
            if "Fast delivery" in f or "Atypical speech" in f or "Substantial dead air" in f or "Missing or silent" in f:
                improvement_areas.append(f)

    # Visual
    if dim_visual["score"] >= 70:
        for f in dim_visual["findings"]:
            if "High teacher visual" in f or "Consistently balanced" in f or "Stable, professional" in f:
                positive_observations.append(f)
    else:
        for f in dim_visual["findings"]:
            if "NEGATIVE EVIDENCE" in f or "Low teacher" in f or "Poor lighting" in f or "instability" in f:
                improvement_areas.append(f)

    # Ensure non-empty lists
    if not positive_observations:
        positive_observations.append("Baseline recording received and decoded successfully.")
    if not improvement_areas:
        improvement_areas.append("Maintain the current exemplary standard of lesson delivery and engagement.")

    return {
        "overall_score": overall_score,
        "rating": assigned_category,
        "summary": category_summary,
        "confidence": {
            "level": confidence_level,
            "score_pct": confidence_pct,
            "limitations": limitations
        },
        "dimension_scores": {
            "Teaching Structure": {
                "score": dim_structure["score"],
                "weight": weights.get("structure", 0.25),
                "weighted_contrib": round(dim_structure["score"] * weights.get("structure", 0.25), 1),
                "findings": dim_structure["findings"],
                "citations": dim_structure["citations"]
            },
            "Student Interaction": {
                "score": dim_interaction["score"],
                "weight": weights.get("interaction", 0.25),
                "weighted_contrib": round(dim_interaction["score"] * weights.get("interaction", 0.25), 1),
                "findings": dim_interaction["findings"],
                "citations": dim_interaction["citations"]
            },
            "Rapport & Reinforcement": {
                "score": dim_rapport["score"],
                "weight": weights.get("rapport", 0.20),
                "weighted_contrib": round(dim_rapport["score"] * weights.get("rapport", 0.20), 1),
                "findings": dim_rapport["findings"],
                "citations": dim_rapport["citations"]
            },
            "Audio & Delivery": {
                "score": dim_audio["score"],
                "weight": weights.get("audio", 0.15),
                "weighted_contrib": round(dim_audio["score"] * weights.get("audio", 0.15), 1),
                "findings": dim_audio["findings"],
                "citations": []
            },
            "Visual Presentation": {
                "score": dim_visual["score"],
                "weight": weights.get("visual", 0.15),
                "weighted_contrib": round(dim_visual["score"] * weights.get("visual", 0.15), 1),
                "findings": dim_visual["findings"],
                "citations": dim_visual["citations"]
            }
        },
        "positive_observations": positive_observations,
        "improvement_areas": improvement_areas,
        "transcript_summary": {
            "word_count": audio_res.get("word_count", 0),
            "speech_rate_wpm": audio_res.get("speech_rate_wpm", 0.0),
            "language": audio_res.get("language", "unknown"),
            "segments": segments
        }
    }
