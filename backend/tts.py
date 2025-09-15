import os
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

def synthesize_audio(text: str, voice: str = "en-US_AllisonV3Voice") -> bytes:
    """
    Convert text to audio using IBM Watson TTS
    """
    authenticator = IAMAuthenticator(os.getenv("IBM_TTS_APIKEY"))
    tts = TextToSpeechV1(authenticator=authenticator)
    tts.set_service_url(os.getenv("IBM_TTS_URL"))

    response = tts.synthesize(
        text,
        voice=voice,
        accept="audio/mp3"
    ).get_result().content

    return response
