import gradio as gr
from faster_whisper import WhisperModel
from transformers import pipeline

# Load FasterWhisper Model
model = WhisperModel("base", device="cpu", compute_type="int8")

# Load Hugging Face Sentiment Analysis Pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Function for real-time transcription and sentiment analysis
def transcribe_and_analyze(audio):
    # Transcribe audio using FasterWhisper
    segments, _ = model.transcribe(audio, beam_size=5)
    
    full_transcription = ""
    sentiment_summary = ""

    for segment in segments:
        # Extract each sentence/phrase in real time
        sentence = segment.text.strip()
        full_transcription += sentence + " "

        # Perform Sentiment Analysis on the completed sentence
        sentiment_result = sentiment_analyzer(sentence)[0]
        sentiment = sentiment_result["label"]
        confidence = sentiment_result["score"]

        # Print the transcription and sentiment analysis in the terminal
        print(f"Transcribed: {sentence}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})")

        # Add sentiment result to the output
        sentiment_summary += f"Sentence: '{sentence}' \nSentiment: {sentiment} (Confidence: {confidence:.2f})\n\n"
    
    return full_transcription, sentiment_summary

# Gradio Interface for live transcription and sentiment analysis
gr.Interface(
    fn=transcribe_and_analyze, 
    inputs=gr.Audio(type="filepath"), 
    outputs=[
        "textbox",   # Transcription output
        "textbox"    # Sentiment analysis output
    ],
    live=True
).launch(share=True)
