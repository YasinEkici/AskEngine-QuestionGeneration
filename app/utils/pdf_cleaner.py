import re

def repair_pdf_text(text: str) -> str:
    """
    Cleans and repairs common PDF text extraction artifacts.
    """
    if not text:
        return ""
        
    # 1. Hyphenation Repair
    # Matches: word ending with hyphen, newline, next line starts with lowercase
    # e.g. "exa-\nmple" -> "example"
    # Basic regex: (\w+)-\n(\w+) -> \1\2
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # 2. Normalize Whitespace
    # Replace multiple spaces/tabs/newlines with single space
    # Exception: We might want to keep paragraph breaks. 
    # Let's keep double newlines as paragraph breaks.
    
    # Strategy: 
    # a. Join broken lines (single newlines) -> space
    # b. Keep double newlines (\n\n) -> \n\n
    
    # Simple approach for QG context: mostly we want a flow of text.
    # Replace (not \n\n) \n with space?
    
    # Better: collapse all whitespace sequences to single space for pure NLP context
    # But for "Context Window" display, keeping paragraphs looks nicer.
    
    # Let's do:
    # - Hyphen fix (done)
    # - Replace non-paragraph newlines with space
    
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # - Collapse multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()
