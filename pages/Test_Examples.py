# import streamlit as st
# import os
# import tempfile
# import glob
# import requests
# from pydub import AudioSegment
# from speechbrain.pretrained import SpectralMaskEnhancement
# from speechbrain.pretrained.interfaces import foreign_class
# import yt_dlp
# import torch
# import torchaudio

# # Set page config
# st.set_page_config(page_title="Test Samples | Accent Classifier", layout="centered")

# # Load model and enhancer once
# @st.cache_resource
# def load_classifier():
#     return foreign_class(
#         source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
#         pymodule_file="custom_interface.py",
#         classname="CustomEncoderWav2vec2Classifier"
#     )

# @st.cache_resource
# def load_enhancer():
#     return SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank")

# classifier = load_classifier()
# enhancer = load_enhancer()

# # Sample audio and video links
# sample_links = {
#     "Indian English (YouTube)": "https://www.youtube.com/watch?v=PI709plS2uk",
#     "British English (YouTube)": "https://www.youtube.com/watch?v=MBzR76Vf0x8",
#     "Indian English (MP3)": "https://www.example.com/indian_sample.mp3",
#     "Australian English (MP3)": "https://www.example.com/australian_sample.mp3",
# }

# # Utility functions
# def download_video(url, out_path_dir):
#     if "youtube.com" in url or "youtu.be" in url:
#         output_template = os.path.join(out_path_dir, "yt_audio.%(ext)s")
#         ydl_opts = {
#             'format': 'bestaudio/best',
#             'outtmpl': output_template,
#             'quiet': True,
#             'noplaylist': True,
#             'postprocessors': [{
#                 'key': 'FFmpegExtractAudio',
#                 'preferredcodec': 'mp3',
#                 'preferredquality': '192',
#             }],
#         }
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             ydl.download([url])
#         downloaded_files = glob.glob(os.path.join(out_path_dir, "yt_audio.*"))
#         if not downloaded_files:
#             raise FileNotFoundError("Audio download from YouTube failed.")
#         return downloaded_files[0]
#     else:
#         response = requests.get(url, stream=True)
#         content_type = response.headers.get("Content-Type", "")
#         if "video" not in content_type and "audio" not in content_type:
#             raise ValueError(f"Unsupported content type: {content_type}")
#         ext = content_type.split('/')[-1]
#         out_path = os.path.join(out_path_dir, f"downloaded.{ext}")
#         with open(out_path, 'wb') as f:
#             for chunk in response.iter_content(8192):
#                 f.write(chunk)
#         return out_path

# def convert_to_wav(input_path, output_path):
#     sound = AudioSegment.from_file(input_path)
#     sound = sound.set_channels(1).set_frame_rate(16000)
#     sound.export(output_path, format="wav")

# def denoise_wav(input_path, output_path):
#     # Load audio
#     signal = enhancer.load_audio(input_path)  # [time]
#     signal = signal.unsqueeze(0)              # [1, time]
#     lengths = torch.tensor([1.0])             # assume full length

#     # Enhance
#     enhanced = enhancer.enhance_batch(signal, lengths)  # [1, time]

#     # Save (needs [channels, time] format)
#     enhanced = enhanced.squeeze(0).unsqueeze(0)  # [1, time] -> [1, time] (explicit 2D)
#     torchaudio.save(output_path, enhanced, 16000)

# def classify(wav_path):
#     out_prob, score, index, text_lab = classifier.classify_file(wav_path)
#     return text_lab[0], float(score) * 100

# # UI layout
# st.title("🔊 Test Samples for Accent Detection")

# choice = st.selectbox("Choose a test sample:", list(sample_links.keys()))
# selected_url = sample_links[choice]
# st.markdown(f"**Selected URL:** [{selected_url}]({selected_url})")

# if st.button("Classify Selected Sample"):
#     with tempfile.TemporaryDirectory() as tmpdir:
#         st.info("Downloading sample...")
#         try:
#             input_path = download_video(selected_url, tmpdir)
#         except Exception as e:
#             st.error(f"Download failed: {e}")
#             st.stop()

#         wav_path = os.path.join(tmpdir, "converted.wav")
#         st.info("Converting to WAV...")
#         try:
#             convert_to_wav(input_path, wav_path)
#         except Exception as e:
#             st.error(f"Conversion failed: {e}")
#             st.stop()

#         denoised_path = os.path.join(tmpdir, "denoised.wav")
#         st.info("Denoising audio...")
#         try:
#             denoise_wav(wav_path, denoised_path)
#         except Exception as e:
#             st.error(f"Denoising failed: {e}")
#             st.stop()

#         st.info("Classifying...")
#         try:
#             accent, confidence = classify(denoised_path)
#             st.success(f"Predicted Accent: **{accent.capitalize()}**")
#             st.metric("Confidence", f"{confidence:.2f}%")
#         except Exception as e:
#             st.error(f"Classification failed: {e}")

# import streamlit as st
# import os
# import tempfile
# import glob
# import requests
# from pydub import AudioSegment
# from speechbrain.pretrained import SpectralMaskEnhancement
# from speechbrain.pretrained.interfaces import foreign_class
# import yt_dlp
# import torch
# import torchaudio

# # Set page config
# st.set_page_config(page_title="Test Samples | Accent Classifier", layout="centered")

# # Load model and enhancer once
# @st.cache_resource
# def load_classifier():
#     return foreign_class(
#         source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
#         pymodule_file="custom_interface.py",
#         classname="CustomEncoderWav2vec2Classifier"
#     )

# @st.cache_resource
# def load_enhancer():
#     return SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank")

# classifier = load_classifier()
# enhancer = load_enhancer()

# # Define sample links (both remote and local)
# sample_links = {
#     "Indian English (YouTube)": "https://www.youtube.com/watch?v=PI709plS2uk",
#     "British English (YouTube)": "https://www.youtube.com/watch?v=MBzR76Vf0x8",
#     "Malaysian English (Local MP3)": "demo_data/malaysia_1.wav",
#     "Australian English (Local MP3)": "demo_data/australian_sample.mp3",
# }

# # Utility functions
# def download_video(url, out_path_dir):
#     if "youtube.com" in url or "youtu.be" in url:
#         output_template = os.path.join(out_path_dir, "yt_audio.%(ext)s")
#         ydl_opts = {
#             'format': 'bestaudio/best',
#             'outtmpl': output_template,
#             'quiet': True,
#             'noplaylist': True,
#             'postprocessors': [{
#                 'key': 'FFmpegExtractAudio',
#                 'preferredcodec': 'mp3',
#                 'preferredquality': '192',
#             }],
#         }
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             ydl.download([url])
#         downloaded_files = glob.glob(os.path.join(out_path_dir, "yt_audio.*"))
#         if not downloaded_files:
#             raise FileNotFoundError("Audio download from YouTube failed.")
#         return downloaded_files[0]
#     else:
#         response = requests.get(url, stream=True)
#         content_type = response.headers.get("Content-Type", "")
#         if "video" not in content_type and "audio" not in content_type:
#             raise ValueError(f"Unsupported content type: {content_type}")
#         ext = content_type.split('/')[-1]
#         out_path = os.path.join(out_path_dir, f"downloaded.{ext}")
#         with open(out_path, 'wb') as f:
#             for chunk in response.iter_content(8192):
#                 f.write(chunk)
#         return out_path

# def convert_to_wav(input_path, output_path):
#     sound = AudioSegment.from_file(input_path)
#     sound = sound.set_channels(1).set_frame_rate(16000)
#     sound.export(output_path, format="wav")

# def denoise_wav(input_path, output_path):
#     signal = enhancer.load_audio(input_path)  # [time]
#     signal = signal.unsqueeze(0)              # [1, time]
#     lengths = torch.tensor([1.0])             # full length
#     enhanced = enhancer.enhance_batch(signal, lengths)  # [1, time]
#     enhanced = enhanced.squeeze(0).unsqueeze(0)         # [1, time] -> [1, time] -> [1, time] for torchaudio
#     torchaudio.save(output_path, enhanced, 16000)

# def classify(wav_path):
#     out_prob, score, index, text_lab = classifier.classify_file(wav_path)
#     return text_lab[0], float(score) * 100

# # UI layout
# st.title("🔊 Test Samples for Accent Detection")

# choice = st.selectbox("Choose a test sample:", list(sample_links.keys()))
# selected_url = sample_links[choice]
# st.markdown(f"**Selected Source:** `{selected_url}`")

# is_local = selected_url.startswith("demo_data/")

# if st.button("Classify Selected Sample"):
#     with tempfile.TemporaryDirectory() as tmpdir:
#         st.info("Getting sample...")

#         try:
#             if is_local:
#                 input_path = os.path.abspath(selected_url)
#                 if input_path.endswith(".mp3"):
#                     st.audio(open(input_path, 'rb').read(), format='audio/mp3')
#             else:
#                 input_path = download_video(selected_url, tmpdir)
#         except Exception as e:
#             st.error(f"Download failed: {e}")
#             st.stop()

#         wav_path = os.path.join(tmpdir, "converted.wav")
#         st.info("Converting to WAV...")
#         try:
#             convert_to_wav(input_path, wav_path)
#         except Exception as e:
#             st.error(f"Conversion failed: {e}")
#             st.stop()

#         denoised_path = os.path.join(tmpdir, "denoised.wav")
#         st.info("Denoising audio...")
#         try:
#             denoise_wav(wav_path, denoised_path)
#         except Exception as e:
#             st.error(f"Denoising failed: {e}")
#             st.stop()

#         st.info("Classifying...")
#         try:
#             accent, confidence = classify(denoised_path)
#             st.success(f"Predicted Accent: **{accent.capitalize()}**")
#             st.metric("Confidence", f"{confidence:.2f}%")
#         except Exception as e:
#             st.error(f"Classification failed: {e}")


import streamlit as st
import os
import tempfile
import glob
import requests
from pydub import AudioSegment
from speechbrain.pretrained import SpectralMaskEnhancement
from speechbrain.pretrained.interfaces import foreign_class
import yt_dlp
import torch
import torchaudio

# Set page config
st.set_page_config(page_title="Test Samples | Accent Classifier", layout="centered")

# Load model and enhancer once
@st.cache_resource
def load_classifier():
    return foreign_class(
        source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier"
    )

@st.cache_resource
def load_enhancer():
    return SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank")

classifier = load_classifier()
enhancer = load_enhancer()


# Utility functions
def download_video(url, out_path_dir):
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
        ext = content_type.split('/')[-1]
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
    signal = enhancer.load_audio(input_path)  # [time]
    signal = signal.unsqueeze(0)              # [1, time]
    lengths = torch.tensor([1.0])             # full length
    enhanced = enhancer.enhance_batch(signal, lengths)  # [1, time]
    enhanced = enhanced.squeeze(0).unsqueeze(0)         # [1, time] -> [1, time] -> [1, time] for torchaudio
    torchaudio.save(output_path, enhanced, 16000)


def classify(wav_path):
    out_prob, score, index, text_lab = classifier.classify_file(wav_path)
    return text_lab[0], float(score) * 100


# YouTube + Local test samples
sample_links = {
    "Indian English (YouTube)": "https://www.youtube.com/watch?v=PI709plS2uk",
    "British English (YouTube)": "https://www.youtube.com/watch?v=0BU_u8_blss",
}

# Load local .mp3 and .wav files from demo_data folder
demo_folder = "demo_data"
local_samples = [
    f for f in os.listdir(demo_folder)
    if f.lower().endswith((".mp3", ".wav"))
]
local_paths = {f"Local File: {name}": os.path.join(demo_folder, name) for name in local_samples}

# Combine
all_samples = {**sample_links, **local_paths}

# UI
st.title("🔊 Test Samples for Accent Detection")

choice = st.selectbox("Choose a test sample:", list(all_samples.keys()))
selected_path = all_samples[choice]
is_local = selected_path.startswith(demo_folder)

# Show link if it's YouTube
if not is_local:
    st.markdown(f"**Selected URL:** [{selected_path}]({selected_path})")

# Play audio if it's .mp3 or .wav
if is_local and selected_path.endswith((".mp3", ".wav")):
    st.markdown("**Preview Audio:**")
    with open(selected_path, 'rb') as f:
        file_bytes = f.read()
        st.audio(file_bytes, format='audio/mp3' if selected_path.endswith(".mp3") else 'audio/wav')

if st.button("Classify Selected Sample"):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download or copy the file
        if is_local:
            input_path = os.path.join(tmpdir, os.path.basename(selected_path))
            with open(selected_path, "rb") as src, open(input_path, "wb") as dst:
                dst.write(src.read())
        else:
            st.info("Downloading sample...")
            try:
                input_path = download_video(selected_path, tmpdir)
            except Exception as e:
                st.error(f"Download failed: {e}")
                st.stop()

        # Convert to WAV
        wav_path = os.path.join(tmpdir, "converted.wav")
        st.info("Converting to WAV...")
        try:
            convert_to_wav(input_path, wav_path)
        except Exception as e:
            st.error(f"Conversion failed: {e}")
            st.stop()

        # Denoise
        denoised_path = os.path.join(tmpdir, "denoised.wav")
        st.info("Denoising audio...")
        try:
            denoise_wav(wav_path, denoised_path)
        except Exception as e:
            st.error(f"Denoising failed: {e}")
            st.stop()

        # Classification
        st.info("Classifying...")
        try:
            accent, confidence = classify(denoised_path)
            st.success(f"Predicted Accent: **{accent.capitalize()}**")
            st.metric("Confidence", f"{confidence:.2f}%")
        except Exception as e:
            st.error(f"Classification failed: {e}")

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
