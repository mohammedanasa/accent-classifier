import streamlit as st
import os
import tempfile
import requests
from moviepy import VideoFileClip
from pydub import AudioSegment
from speechbrain.pretrained.interfaces import foreign_class
import yt_dlp
import glob
from speechbrain.pretrained import SpectralMaskEnhancement
import torch
import torchaudio


# Page config
st.set_page_config(page_title="English Accent Classifier", layout="centered")

# Load models once
@st.cache_resource
def load_model():
    return foreign_class(
        source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier"
    )

@st.cache_resource
def load_denoiser():
    return SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank")

classifier = load_model()
enhancer = load_denoiser()

# Utils
def download_video(url, out_path_dir):
    """
    Downloads audio from a YouTube link or direct URL.
    Returns the full path to the downloaded file.
    """
    if "youtube.com" in url or "youtu.be" in url:
        output_template = os.path.join(out_path_dir, "yt_audio.%(ext)s")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'quiet': True,
            'noplaylist': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        downloaded_files = glob.glob(os.path.join(out_path_dir, "yt_audio.*"))
        if not downloaded_files:
            raise FileNotFoundError("Audio download from YouTube failed.")
        return downloaded_files[0]
    else:
        response = requests.get(url, stream=True)
        content_type = response.headers.get("Content-Type", "")
        if "video" not in content_type and "audio" not in content_type:
            raise ValueError(f"Unsupported content type: {content_type}")
        ext = content_type.split('/')[-1].split(";")[0]
        out_path = os.path.join(out_path_dir, f"downloaded.{ext}")
        with open(out_path, 'wb') as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)
        return out_path

def convert_to_wav(input_path, output_path):
    sound = AudioSegment.from_file(input_path)
    sound = sound.set_channels(1).set_frame_rate(16000)
    sound.export(output_path, format="wav")

def denoise_wav(input_path, output_path):
    # Load audio
    signal = enhancer.load_audio(input_path)  # [time]
    signal = signal.unsqueeze(0)              # [1, time]
    lengths = torch.tensor([1.0])             # assume full length

    # Enhance
    enhanced = enhancer.enhance_batch(signal, lengths)  # [1, time]

    # Save (needs [channels, time] format)
    enhanced = enhanced.squeeze(0).unsqueeze(0)  # [1, time] -> [1, time] (explicit 2D)
    torchaudio.save(output_path, enhanced, 16000)


def classify(wav_path):
    out_prob, score, index, text_lab = classifier.classify_file(wav_path)
    return text_lab[0], float(score) * 100

# UI
st.title("🎙️ English Accent Classifier")

input_mode = st.radio("Choose input method:", ["Upload audio file", "Use YouTube or video URL"])
uploaded_file = None
url = ""

if input_mode == "Upload audio file":
    uploaded_file = st.file_uploader("Upload MP3 or WAV", type=["mp3", "wav"])
else:
    url = st.text_input("Enter YouTube or direct video URL")

if uploaded_file or url:
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            if uploaded_file:
                ext = uploaded_file.name.split('.')[-1]
                input_path = os.path.join(tmpdir, f"input.{ext}")
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.read())
            else:
                st.info("Downloading from URL...")
                input_path = download_video(url, tmpdir)

            wav_path = os.path.join(tmpdir, "converted.wav")
            st.info("Converting to WAV...")
            convert_to_wav(input_path, wav_path)

            denoised_path = os.path.join(tmpdir, "denoised.wav")
            st.info("Enhancing audio (denoising)...")
            denoise_wav(wav_path, denoised_path)

            st.info("Running model inference...")
            accent, confidence = classify(denoised_path)

            st.success(f"Predicted Accent: **{accent.capitalize()}**")
            st.metric("Confidence", f"{confidence:.2f}%")
        except Exception as e:
            st.error(f"Something went wrong: {e}")


# Footer Warning
st.markdown(
    """
    <hr>
    <div style='text-align: center; font-size: 0.9em; color: #ff4b4b'>
        ⚠️ <strong>Note:</strong> Some videos might take longer to process. <br>
        For best performance, please use **shorter videos or audio clips** due to GPU limitations. <br>
        Thank you for your patience!
    </div>
    """,
    unsafe_allow_html=True
)


# Footer
st.markdown(
    """
    <hr>
    <div style='text-align: center'>
        Created by Mohammed Anas <br>
        <a href='https://www.linkedin.com/in/mohammedanasa/' target='_blank'>Linkedin</a> ·
        <a href='https://github.com/mohammedanasa/accent-detector-app' target='_blank'>GitHub Repo</a>
    </div>
    """,
    unsafe_allow_html=True
)