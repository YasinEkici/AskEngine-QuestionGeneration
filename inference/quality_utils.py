import re
from collections import Counter

def evaluate_quality(question: str, answer_text: str = "") -> dict:
    """
    Evaluates the quality of a generated question based on heuristic rules.
    Returns a dict with 'is_accepted', 'score', and 'reason'.
    """
    if not question:
        return {"is_accepted": False, "score": 0.0, "reason": "Empty"}
        
    # 1. Minimum Length (User Rule: 10 chars)
    if len(question) < 10:
        return {"is_accepted": False, "score": 0.1, "reason": "Too short (<10 chars)"}
        
    # 2. Format: Must end with '?'
    # Soft check: If valid but missing '?', we can append it, but strict filter might reject.
    # Let's enforce strictness for high quality, or just penalize score.
    # Decision: Penalize score, don't reject outright if it looks okay otherwise.
    if not question.strip().endswith("?"):
         # Check if it ends with punctuation that isn't ?
        if question.strip()[-1] in "!.,":
             return {"is_accepted": False, "score": 0.4, "reason": "Invalid punctuation"}
        # If no punctuation, it might be truncated.
    
    # 3. Forbidden Patterns (Prompt Leaks)
    forbidden = ["context:", "answer:", "question:", "answer is"]
    lower_q = question.lower()
    for pattern in forbidden:
        if pattern in lower_q:
            return {"is_accepted": False, "score": 0.0, "reason": f"Prompt leak detected ('{pattern}')"}

    # 4. Answer Leakage (Answer appears verbatim in question)
    # Only if answer is roughly long enough to matter (e.g. > 4 chars)
    if answer_text and len(answer_text) > 4:
        if answer_text.lower() in lower_q:
            return {"is_accepted": False, "score": 0.3, "reason": "Answer contained in question"}

    # 5. Repetition Check
    words = lower_q.split()
    counts = Counter(words)
    # Ignore common stopwords
    common_stopwords = {"the", "a", "an", "is", "of", "to", "in", "what"}
    for word, count in counts.items():
        if count >= 3 and word not in common_stopwords:
             return {"is_accepted": False, "score": 0.2, "reason": f"Repetitive word ('{word}')"}

    # If pass all
    return {"is_accepted": True, "score": 1.0, "reason": "OK"}
