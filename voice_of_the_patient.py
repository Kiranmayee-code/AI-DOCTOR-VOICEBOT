# Import necessary libraries
import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import os
from groq import Groq

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Audio Recording Function
def record_audio(file_path, timeout=20, phrase_time_limit=None):
    """
    Records audio from the microphone and saves it as an MP3 file.

    Args:
        file_path (str): Path to save the recorded audio file.
        timeout (int): Maximum time to wait for a phrase to start (in seconds).
        phrase_time_limit (int): Maximum time for the phrase to be recorded (in seconds).
    """
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Start speaking now...")
            
            # Record the audio
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")
            
            # Convert the recorded audio to an MP3 file
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            
            logging.info(f"Audio saved to {file_path}")

    except Exception as e:
        logging.error(f"An error occurred while recording: {e}")

# Step 2: Speech-to-Text Transcription Function
def transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY):
    """
    Transcribes an audio file using the Groq API.

    Args:
        stt_model (str): Name of the speech-to-text model.
        audio_filepath (str): Path to the audio file.
        GROQ_API_KEY (str): Groq API key for authentication.
    """
    try:
        # Check if the audio file exists
        if not os.path.exists(audio_filepath):
            raise FileNotFoundError(f"Audio file not found at {audio_filepath}")
        
        client = Groq(api_key=GROQ_API_KEY)
        
        # Open the audio file
        with open(audio_filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                language="en"
            )
        
        logging.info("Transcription complete.")
        return transcription.text

    except Exception as e:
        logging.error(f"An error occurred while transcribing: {e}")
        return None

# Main Functionality
if __name__ == "__main__":
    # Define file path and Groq API key
    audio_filepath = "patient_voice_test_for_patient.mp3"
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # Replace with your API key if not set in environment
    stt_model = "whisper-large-v3-turbo"

    # Record audio
    logging.info("Recording audio...")
    record_audio(file_path=audio_filepath)

    # Transcribe audio
    logging.info("Transcribing audio...")
    transcription = transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY)

    # Print transcription
    if transcription:
        logging.info(f"Transcription result: {transcription}")
    else:
        logging.error("Failed to transcribe the audio.")