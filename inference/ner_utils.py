import re
from collections import Counter

def extract_key_phrases(text: str, top_k: int = 5) -> list[str]:
    """
    Extracts potential answer candidates using heuristic rules (Regex).
    Focuses on Proper Nouns (Capitalized), Dates, and Numbers.
    """
    if not text:
        return []
        
    candidates = []
    
    # 1. Dates (Years 1000-2999)
    dates = re.findall(r'\b(19|20)\d{2}\b', text)
    # The regex above captures the group (19|20). We want the full match.
    # Correct regex for full match year:
    date_matches = re.finditer(r'\b((?:19|20)\d{2})\b', text)
    for m in date_matches:
        candidates.append(m.group(0))

    # 2. Capitalized Phrases (Proper Nouns)
    # Matches sequence of Capitalized words (e.g. "Neil Armstrong", "United States")
    # Avoid sentence starters if possible? Hard without NLP.
    # Just grab them all, maybe filter simple stop words later.
    # Regex: (Upper[a-z]+ (Space Upper[a-z]+)*)
    cap_phrases = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
    
    # Filter out common sentence starters if they appear at start of lines? 
    # For now, just add them. Heuristics are fuzzy.
    # Filter single words that are likely stopwords "The", "A", "In"
    stopwords = {"The", "A", "An", "In", "On", "At", "To", "For", "Of", "With", "By", "And", "But", "Or", "So", "However", "Therefore"}
    
    for phrase in cap_phrases:
        if phrase in stopwords:
            continue
        if len(phrase) < 3: # Skip "Al", "Bo" etc unless useful?
            continue
        candidates.append(phrase)

    # 3. Numeric stats (e.g. "380,000", "50%")
    # Matches digits with optional commas/decimals, maybe followed by %
    numerics = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?\b', text)
    # Filter small integers like 1, 2, 3 unless they look special? 
    # Let's keep them if they are not year-like (handled above)
    for num in numerics:
        if len(num) > 1: # Skip single digits
             if not re.match(r'(19|20)\d{2}', num): # Avoid dups with years
                 candidates.append(num)

    # 4. Ranking
    # Count frequency
    counts = Counter(candidates)
    
    # Sort by Frequency then Length (Longer phrases usually better answers)
    # We want unique list.
    sorted_candidates = sorted(counts.keys(), key=lambda x: (counts[x], len(x)), reverse=True)
    
    return sorted_candidates[:top_k]
