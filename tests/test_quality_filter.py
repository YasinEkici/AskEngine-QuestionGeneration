from inference.quality_utils import evaluate_quality

def test_too_short_question():
    q = "Why?"
    res = evaluate_quality(q)
    assert res["is_accepted"] is False
    assert "short" in res["reason"]

def test_no_question_mark_is_allowed_if_long_enough_but_maybe_penalized_or_accepted():
    # Based on our implementation, missing ? is not an immediate reject unless it ends in ., ! or similar invalid punctuation for a question? 
    # Wait, code says: if not ends with ?, check punctuation.
    q = "What is the capital of France" # No punctuation
    res = evaluate_quality(q)
    # Our code: if not question.strip().endswith("?"): ...
    # if valid punctuation -> reject. if no punctuation (truncated) -> allowed?
    # Let's check code: "if question.strip()[-1] in '!.,': return False"
    # "e" is not in "!.," so it should pass (maybe).
    # But wait, we wanted Robustness. Let's verify strictness.
    # Actually current implementation allows it if it doesn't end in bad punctuation.
    assert res["is_accepted"] is True 

def test_prompt_leak():
    q = "answer: Paris context: France"
    res = evaluate_quality(q)
    assert res["is_accepted"] is False
    assert "leak" in res["reason"]

def test_answer_leak():
    q = "The capital is Paris?"
    ans = "Paris"
    res = evaluate_quality(q, ans)
    # "Paris" in "The capital is Paris?" -> True
    assert res["is_accepted"] is False
    assert "Answer contained" in res["reason"]

def test_repetition():
    q = "Generative AI Generative AI Generative AI is cool?"
    res = evaluate_quality(q)
    assert res["is_accepted"] is False
    assert "Repetitive" in res["reason"]
