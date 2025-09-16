# backend/tts.py
import io
import time
from typing import List, Optional

from gtts import gTTS
from pydub import AudioSegment

from .utils import chunk_text_by_chars


class TTSException(RuntimeError):
    """Raised for TTS synthesis related errors."""


def _synthesize_chunk_to_segment(text_chunk: str, lang: str, tld: str, attempt: int = 1) -> AudioSegment:
    """
    Synthesize a single chunk to a pydub.AudioSegment.
    Retries should be handled by the caller if desired.
    """
    try:
        tts = gTTS(text=text_chunk, lang=lang, tld=tld, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        seg = AudioSegment.from_file(buf, format="mp3")
        return seg
    except Exception as e:
        raise TTSException(f"gTTS failed for chunk (attempt {attempt}): {e}") from e


def synthesize_gtts_bytes(
    text: str,
    lang: str = "en",
    tld: str = "com",
    max_chunk_chars: int = 3000,
    chunk_retries: int = 2,
    backoff_base: float = 0.8,
    pause_between_chunks: float = 0.12,
) -> bytes:
    """
    Convert text -> MP3 bytes using gTTS, chunking and concatenating via pydub.

    Raises:
        TTSException on irrecoverable failure.

    Args:
        text: input text to synthesize.
        lang: gTTS language code (default 'en').
        tld: gTTS TLD (e.g., 'com', 'co.uk').
        max_chunk_chars: maximum characters per chunk (to avoid gTTS/HTTP limits).
        chunk_retries: how many attempts per chunk before giving up.
        backoff_base: base seconds for exponential backoff between retries.
        pause_between_chunks: short sleep between successful chunk requests to reduce rate issues.

    Returns:
        bytes: MP3 data.
    """
    if text is None:
        raise TTSException("No text provided for synthesis.")
    text = text.strip()
    if not text:
        raise TTSException("Input text is empty after trimming.")

    chunks: List[str] = chunk_text_by_chars(text, max_chars=max_chunk_chars)
    if not chunks:
        raise TTSException("No chunks were produced from input text.")

    segments: List[AudioSegment] = []

    for idx, chunk in enumerate(chunks):
        last_exc: Optional[Exception] = None
        for attempt in range(1, chunk_retries + 1):
            try:
                seg = _synthesize_chunk_to_segment(chunk, lang=lang, tld=tld, attempt=attempt)
                segments.append(seg)
                # small polite pause between successful chunk requests
                if pause_between_chunks:
                    time.sleep(pause_between_chunks)
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                # exponential backoff before retrying
                sleep_for = backoff_base * (2 ** (attempt - 1))
                time.sleep(sleep_for)
                continue
        if last_exc is not None:
            # include chunk index and number of attempts in error
            raise TTSException(f"Failed to synthesize chunk #{idx} after {chunk_retries} attempts: {last_exc}") from last_exc

    if not segments:
        raise TTSException("gTTS produced no audio segments.")

    # Concatenate segments
    combined = segments[0]
    for seg in segments[1:]:
        combined += seg

    # Export combined audio to mp3 bytes
    try:
        out_buf = io.BytesIO()
        combined.export(out_buf, format="mp3")
        out_buf.seek(0)
        return out_buf.read()
    except Exception as e:
        raise TTSException(f"Failed to export combined audio to mp3: {e}") from e
