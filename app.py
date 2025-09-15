import streamlit as st
from backend.rewrite import rewrite_text
from backend.tts import synthesize_audio

st.set_page_config(page_title="EchoVerse - MVP", layout="wide")
st.title("🎙 EchoVerse - AI Audiobook Creator (Phase 1 MVP)")

# Input
text = st.text_area("Enter text here", height=200)
tone = st.selectbox("Choose Tone", ["Neutral", "Motivational", "Suspenseful"])
voice = st.selectbox("Choose Voice", ["en-US_AllisonV3Voice", "en-US_MichaelV3Voice", "en-US_LisaV3Voice"])

if st.button("Generate Narration"):
    if text.strip():
        # Rewrite
        rewritten = rewrite_text(text, tone)
        st.subheader("Rewritten")
        st.write(rewritten)

        # Synthesize Audio
        audio_bytes = synthesize_audio(rewritten, voice)

        # Playback + Download
        st.audio(audio_bytes, format="audio/mp3")
        st.download_button(
            "Download MP3",
            audio_bytes,
            file_name="echoverse_output.mp3",
            mime="audio/mpeg"
        )
    else:
        st.warning("Please enter some text before generating.")
