# Import necessary libraries
from dotenv import load_dotenv
load_dotenv()

import os
import gradio as gr
import logging
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import record_audio, transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_elevenlabs
import pygame  # Import pygame for MP3 playback

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the system prompt
system_prompt = """You have to act as a professional doctor, I know you are not but this is for learning purposes. 
What's in this image? Do you find anything wrong with it medically? 
If you make a differential, suggest some remedies for them. Do not add any numbers or special characters in 
your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person. 
Do not say 'In the image I see' but say 'With what I see, I think you have ....' 
Do not respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot. 
Keep your answer concise (max 2 sentences). No preamble, start your answer right away, please."""

# Define the function to process audio and image inputs
def process_inputs(audio_filepath, image_filepath):
    try:
        # Check if audio file exists
        if not audio_filepath or not os.path.exists(audio_filepath):
            return "No audio file provided", "Unable to process the request due to missing audio input", None

        # Transcribe audio
        speech_to_text_output = transcribe_with_groq(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
            audio_filepath=audio_filepath,
            stt_model="whisper-large-v3-turbo"
        )
        if not speech_to_text_output:
            return "Error during transcription", "Unable to process inputs", None

        # Handle image input
        if image_filepath:
            doctor_response = analyze_image_with_query(
                query=system_prompt + " " + speech_to_text_output,
                encoded_image=encode_image(image_filepath),
                model="meta-llama/llama-4-scout-17b-16e-instruct"  # Updated model
            )
        else:
            doctor_response = "No image provided for analysis."

        # Convert doctor's response to audio
        voice_of_doctor = text_to_speech_with_elevenlabs(
            input_text=doctor_response,
            output_filepath="final.mp3"
        )

        # MP3 playback using pygame
        pygame.mixer.init()
        pygame.mixer.music.load("final.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pass

        return speech_to_text_output, doctor_response, "MP3 played successfully"

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return "Error during processing", "Unable to process inputs", None

# Create the Gradio interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Textbox(label="Doctor's Response in speech")
    ],
    title="AI Doctor with Vision and Voice",
    description="Upload an audio file (or record using your microphone) and an image for analysis. The AI will act as a doctor, analyze the image, and respond to your spoken symptoms."
)

# Launch the interface with debugging enabled
iface.launch(debug=True)