import librosa
import soundfile as sf
import torch
import numpy as np
import os

from transformers import WhisperFeatureExtractor
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor

import subprocess
from flask import Flask, render_template, request
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
app = Flask(__name__)

result = []

AUDIO_PATH = "audio/raw_audio"
DENOISED_AUDIO_PATH = "audio/denoised_audio"

feature_extractor = WhisperFeatureExtractor.from_pretrained("0x2a34/trained_model")
model = WhisperForConditionalGeneration.from_pretrained("0x2a34/trained_model")
processor = WhisperProcessor.from_pretrained("byoussef/whisper-large-v2-Ko", language="Korean", task="transcribe")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

from reduce_noise import reduce_noise


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        file = request.files['audioFile']
        # 받은 데이터를 처리하고 결과를 반환
        if file:
            try:
                # audio path에 저장
                print(file.filename)
                file.save(os.path.join(AUDIO_PATH, file.filename))
                # noise reduction
                if reduce_noise(file.filename):
                    print('noise reduction 완료')
                    file = os.path.join(DENOISED_AUDIO_PATH, file.filename)
                else:
                    print('noise reduction 실패')
                    file = os.path.join(AUDIO_PATH, file.filename)
                result = transcribe(file)
                print('STT 결과:', result)
                return result
            except subprocess.CalledProcessError as e:
                print('STT 코드 실행 오류:', e)
                return 'STT 코드 실행 오류'
    else:
        return render_template("index.html")


def transcribe(file):
    speech_array, sampling_rate = sf.read(file)
    speech_array = librosa.resample(np.asarray(speech_array), orig_sr=sampling_rate, target_sr=16000)

    max_duration = 30  # 30 seconds
    sampling_rate = 16000
    max_samples = max_duration * sampling_rate

    # Split the audio into 30-second segments
    audio_segments = [
        speech_array[i:i + max_samples] for i in range(0, len(speech_array), max_samples)
    ]

    transcriptions = []

    for segment in audio_segments:
        input_features = feature_extractor(segment, sampling_rate=sampling_rate, return_tensors="pt").input_features
        predicted_ids = model.generate(inputs=input_features.to(device))[0]
        transcription = processor.decode(predicted_ids, skip_special_tokens=True)
        transcriptions.append(transcription)

    # Concatenate transcriptions from all segments
    result_transcription = ''.join(transcriptions)

    return result_transcription


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(sys.argv[1]))