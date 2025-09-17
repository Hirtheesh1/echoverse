# app.py
import os
import time
from dotenv import load_dotenv, find_dotenv
import streamlit as st

# Load env
env_path = find_dotenv()
if env_path:
    load_dotenv(env_path)
else:
    load_dotenv(".env")

# backend imports
from backend.rewrite import rewrite_text
from backend.tts import synthesize_gtts_bytes

# UI helpers
def render_latest_ui(latest, dl_key_suffix="latest"):
    st.markdown("---")
    st.header("Original vs Rewritten (Latest)")
    lcol, rcol = st.columns(2)
    with lcol:
        st.subheader("Original")
        st.text_area(
            "Original text (view)",
            latest.get("original", ""),
            height=260,
            key=f"orig_latest_view_{dl_key_suffix}",
            label_visibility="collapsed",
        )
    with rcol:
        st.subheader("Rewritten")
        st.text_area(
            "Rewritten text (view)",
            latest.get("rewritten", ""),
            height=260,
            key=f"rew_latest_view_{dl_key_suffix}",
            label_visibility="collapsed",
        )

    st.markdown("---")
    st.subheader("Listen to your audiobook")
    if latest.get("audio_bytes"):
        try:
            st.audio(latest["audio_bytes"])
        except TypeError:
            st.warning("Audio playback widget failed; use the download button.")
        st.download_button(
            "Download MP3",
            data=latest["audio_bytes"],
            file_name=f"echoverse_narration_{dl_key_suffix}.mp3",
            mime="audio/mpeg",
            key=f"dl_{dl_key_suffix}",
        )
    else:
        st.info("Audio not generated for the latest narration.")


# Page header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align:center;'>🎧 <b>EchoVerse</b></h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color: #333;'>Convert text into expressive audiobooks using Gemini + gTTS</p>", unsafe_allow_html=True)

# Main layout
with st.container():
    st.markdown("## Input")
    left, right = st.columns([2, 1])
    with left:
        uploaded_file = st.file_uploader(
            "Drag and drop a .txt file or browse",
            type=["txt"],
            help="Moderate sized files are fine; long inputs will be chunked.",
        )
        pasted_text = st.text_area("Or paste your text here", height=220, key="paste_input")
    with right:
        st.markdown("### Settings")
        tone = st.selectbox("Select Tone", ["Neutral", "Suspenseful", "Inspiring"], key="tone_select")
        voice_label = st.selectbox("Select Voice (approximate)", ["Lisa", "Michael", "Allison", "Kate"], key="voice_select")
        VOICE_MAP = {
            "Lisa": ("en", "co.uk"),
            "Michael": ("en", "com"),
            "Allison": ("en", "com"),
            "Kate": ("en", "co.uk"),
        }
        lang, tld = VOICE_MAP.get(voice_label, ("en", "com"))
        generate_btn = st.button("🎛️ Generate Audiobook", key="generate_btn")

# Decide input text
input_text = ""
if uploaded_file is not None:
    try:
        raw = uploaded_file.read()
        input_text = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        st.success("File loaded successfully.")
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    input_text = pasted_text

# Initialize session state
if "past_narrations" not in st.session_state:
    st.session_state["past_narrations"] = []

MAX_HISTORY = int(os.getenv("ECHOVERSE_MAX_HISTORY", "6"))

# Handle generation
if generate_btn:
    if not input_text or not input_text.strip():
        st.error("Please paste or upload some text before generating.")
    else:
        # 1) Rewrite with Gemini inside rewrite_text()
        with st.spinner("Rewriting text (inference)..."):
            rewritten, infer_failed, diags = rewrite_text(input_text, tone=tone)

        if infer_failed:
            st.warning("Inference failed on all chunks; using original text for TTS. See diagnostics in console.")
            print("Inference diagnostics:", diags)
        else:
            st.success("Text rewritten.")

        # 2) Synthesize audio with gTTS (catch backend errors)
        audio_bytes = None
        with st.spinner("Synthesizing audio with gTTS..."):
            try:
                audio_bytes = synthesize_gtts_bytes(rewritten, lang=lang, tld=tld)
                if audio_bytes:
                    st.success("Audio synthesized.")
                else:
                    st.warning("Audio generation returned empty bytes. Try shorter input or adjust chunk size.")
            except RuntimeError as rte:
                st.error(f"Audio generation error: {rte}")
                print("gTTS RuntimeError:", str(rte))
                audio_bytes = None
            except Exception as e:
                st.error("Unexpected error during audio synthesis. See console for details.")
                print("Unexpected synthesize_gtts_bytes error:", repr(e))
                audio_bytes = None

        # 3) Save entry to session state (trim history)
        entry = {
            "original": input_text,
            "rewritten": rewritten,
            "tone": tone,
            "voice": voice_label,
            "audio_bytes": audio_bytes,
            "timestamp": time.time(),
            "infer_failed": infer_failed,
            "diags": diags,
        }
        st.session_state.past_narrations.insert(0, entry)
        st.session_state.past_narrations = st.session_state.past_narrations[:MAX_HISTORY]

        # 4) Render latest
        latest = st.session_state.past_narrations[0]
        render_latest_ui(latest, dl_key_suffix="latest")

# Show latest when not generating
if st.session_state.past_narrations and not generate_btn:
    render_latest_ui(st.session_state.past_narrations[0], dl_key_suffix="latest_view")

# Past Narrations panel
st.markdown("---")
st.subheader("Past Narrations (this session)")
if not st.session_state.past_narrations:
    st.info("No narrations yet. Generate one to see it listed here.")
else:
    for idx, item in enumerate(st.session_state.past_narrations):
        label = f"Narration #{len(st.session_state.past_narrations)-idx} — {item.get('tone','')} / {item.get('voice','')}"
        with st.expander(label):
            st.markdown(
                f"**Preview (first 600 chars):**\n\n{item.get('rewritten','')[:600]}{'...' if len(item.get('rewritten',''))>600 else ''}"
            )
            if item.get("audio_bytes"):
                try:
                    st.audio(item["audio_bytes"])
                except TypeError:
                    st.warning("Audio playback widget failed for this entry; you can download it.")
                st.download_button(
                    "Download MP3",
                    data=item["audio_bytes"],
                    file_name=f"narration_{idx+1}.mp3",
                    mime="audio/mpeg",
                    key=f"dl_past_{idx}",
                )
            else:
                st.info("Audio not available for this entry.")

            st.text_area(
                "Original text (past)",
                item.get("original", ""),
                height=160,
                key=f"past_orig_{idx}",
                label_visibility="collapsed",
            )
            st.text_area(
                "Rewritten text (past)",
                item.get("rewritten", ""),
                height=160,
                key=f"past_rew_{idx}",
                label_visibility="collapsed",
            )
