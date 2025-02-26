from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file, send_from_directory
import os
from google.cloud import speech, language_v2
from google.protobuf import wrappers_pb2
from google.cloud import texttospeech_v1

# Initialize Google Cloud clients
tts_client = texttospeech_v1.TextToSpeechClient()
asr_client = speech.SpeechClient()
language_client = language_v2.LanguageServiceClient()

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function for audio-to-text using Google Speech-to-Text
def sample_recognize(content):
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        language_code="en-US",
        model="latest_long",
        audio_channel_count=1,
        enable_word_confidence=True,
        enable_word_time_offsets=True,
    )
    operation = asr_client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=90)

    txt = ''
    for result in response.results:
        txt += result.alternatives[0].transcript + '\n'
    return txt

# Function for text-to-speech using Google Text-to-Speech
def sample_synthesize_speech(text=None, ssml=None):
    input = texttospeech_v1.SynthesisInput(text=text)
    voice = texttospeech_v1.VoiceSelectionParams(language_code="en-US")
    audio_config = texttospeech_v1.AudioConfig(audio_encoding="LINEAR16")
    request = texttospeech_v1.SynthesizeSpeechRequest(input=input, voice=voice, audio_config=audio_config)
    response = tts_client.synthesize_speech(request=request)
    return response.audio_content

# Function to analyze sentiment using Google Cloud Natural Language API
def sample_analyze_sentiment(text_content: str):
    """
    Analyzes Sentiment in a string and returns the sentiment type, score, and magnitude.
    """
    document_type_in_plain_text = language_v2.Document.Type.PLAIN_TEXT
    language_code = "en"
    document = {
        "content": text_content,
        "type_": document_type_in_plain_text,
        "language_code": language_code,
    }
    encoding_type = language_v2.EncodingType.UTF8
    response = language_client.analyze_sentiment(request={"document": document, "encoding_type": encoding_type})
    score = response.document_sentiment.score
    magnitude = response.document_sentiment.magnitude
    sentiment_type = "POSITIVE" if score > 0.75 else "NEGATIVE" if score < -0.75 else "NEUTRAL"
    return sentiment_type, score, magnitude

# Helper function to check allowed files
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to list files
def get_files():
    files = [f for f in os.listdir(UPLOAD_FOLDER) if allowed_file(f)]
    files.sort(reverse=True)
    return files

# Route to display homepage with uploaded files
@app.route('/')
def index():
    files = get_files()
    return render_template('index.html', files=files)

# Handle audio uploads and transcription + sentiment analysis
@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio_data' not in request.files:
        return redirect(request.url)
    file = request.files['audio_data']
    if file and allowed_file(file.filename):
        filename = datetime.now().strftime("%Y%m%d-%I%M%S%p") + '.wav'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform speech-to-text transcription
        with open(file_path, 'rb') as f:
            audio_data = f.read()
        transcription = sample_recognize(audio_data)

        # Perform sentiment analysis on the transcription
        sentiment_type, sentiment_score, sentiment_magnitude = sample_analyze_sentiment(transcription)

        # Save transcription and sentiment result
        with open(file_path + '.txt', 'w') as f:
            f.write(f"Transcription:\n{transcription}\n\n")
            f.write(f"Sentiment: {sentiment_type}\nScore: {sentiment_score}\nMagnitude: {sentiment_magnitude}\n")
    return redirect('/')

# Handle text-to-speech + sentiment analysis
@app.route('/upload_text', methods=['POST'])
def upload_text():
    text = request.form['text']

    # Perform sentiment analysis on the input text
    sentiment_type, sentiment_score, sentiment_magnitude = sample_analyze_sentiment(text)

    # Generate audio from the text
    wav = sample_synthesize_speech(text)
    filename = 'tts_' + datetime.now().strftime("%Y%m%d-%I%M%S%p") + '.wav'
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Save the generated audio
    with open(file_path, 'wb') as f:
        f.write(wav)

    # Save the text and sentiment result
    with open(file_path + '.txt', 'w') as f:
        f.write(f"Text: {text}\n\n")
        f.write(f"Sentiment: {sentiment_type}\nScore: {sentiment_score}\nMagnitude: {sentiment_magnitude}\n")
    return redirect('/')

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route to serve the JavaScript file
@app.route('/script.js', methods=['GET'])
def scripts_js():
    return send_file('./script.js')

if __name__ == '__main__':
    app.run(debug=True, port=8080)
