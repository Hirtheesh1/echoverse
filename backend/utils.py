# backend/utils.py

def chunk_text_by_chars(text, max_chars=3000):
    """
    Chunk text by paragraphs where possible, staying under max_chars per chunk.
    Returns a list of string chunks.
    """
    if not text:
        return []
    paragraphs = text.split("\n\n")
    chunks = []
    cur = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p) if cur else p
        else:
            if cur:
                chunks.append(cur)
            if len(p) <= max_chars:
                cur = p
            else:
                # split long paragraph into smaller slices
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i+max_chars])
                cur = ""
    if cur:
        chunks.append(cur)
    return chunks

def safe_trim(text, max_chars=20000):
    """
    Trim overall input length to a safe maximum. Return an empty string for falsy input.
    Appends a marker if truncated.
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"
