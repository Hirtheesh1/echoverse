# utils/audio_utils.py
import os
import math
import uuid
from typing import List, Tuple
from pydub import AudioSegment

# ---------- CONFIG ----------
TMP_DIR = os.getenv("ECHO_TMP_DIR", "tmp_audio")
os.makedirs(TMP_DIR, exist_ok=True)
# ----------------------------

def chunk_text_by_chars(text: str, max_chars: int = 1500) -> List[str]:
    """
    Simple text chunker by characters with sentence-boundary preference.
    max_chars: approx char size to keep under token limits for LLM/TTS.
    """
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        # try to move end to nearest sentence boundary ('.', '!', '?') backward
        if end < n:
            sep_pos = text.rfind('.', start, end)
            if sep_pos == -1:
                sep_pos = text.rfind('!', start, end)
            if sep_pos == -1:
                sep_pos = text.rfind('?', start, end)
            if sep_pos != -1 and sep_pos > start:
                end = sep_pos + 1
        chunks.append(text[start:end].strip())
        start = end
    return chunks

def export_audiosegment_to_file(seg: AudioSegment, out_path: str):
    """
    Save AudioSegment to an mp3 (ensures bitrate ok).
    """
    seg.export(out_path, format="mp3", bitrate="192k")

def merge_audio_files(file_paths: List[str], out_path: str) -> str:
    """
    Merge list of mp3/wav files into single mp3 using pydub.
    Returns path to merged file.
    """
    if not file_paths:
        raise ValueError("No audio files to merge.")
    combined = AudioSegment.empty()
    for fp in file_paths:
        seg = AudioSegment.from_file(fp)
        combined += seg
    export_audiosegment_to_file(combined, out_path)
    return out_path

def save_bytes_as_file(b: bytes, filename: str) -> str:
    """
    Save bytes (e.g., TTS response) to tmp file and return path.
    """
    os.makedirs(TMP_DIR, exist_ok=True)
    path = os.path.join(TMP_DIR, filename)
    with open(path, "wb") as f:
        f.write(b)
    return path

def create_temp_filename(prefix: str = "audio", ext: str = "mp3") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}.{ext}"

def approximate_sentence_durations(text_chunks: List[str], audio_file_paths: List[str]) -> List[Tuple[str, float]]:
    """
    Return list of (chunk_text, duration_secs) approximated from audio files.
    Useful for sentence-level karaoke highlighting (coarse).
    """
    out = []
    for chunk, path in zip(text_chunks, audio_file_paths):
        seg = AudioSegment.from_file(path)
        duration_s = len(seg) / 1000.0
        out.append((chunk, duration_s))
    return out

def cleanup_tmp_dir(max_keep: int = 50):
    """
    Remove older tmp files if many accumulate.
    """
    files = sorted(
        [os.path.join(TMP_DIR, f) for f in os.listdir(TMP_DIR)],
        key=os.path.getmtime
    )
    if len(files) <= max_keep:
        return
    for f in files[:-max_keep]:
        try:
            os.remove(f)
        except Exception:
            pass
