import streamlit as st

st.set_page_config(page_title="About - English Accent Classifier", layout="centered")
st.title("📘 About the English Accent Classifier")

st.markdown("""
### 🧠 Model Information
- **Model Name**: `accent-id-commonaccent_xlsr-en-english`
- **Authors**: Juan P. Zuluaga et al.
- **Framework**: [SpeechBrain](https://speechbrain.readthedocs.io/en/latest/)
- **Architecture**: wav2vec 2.0 + classifier head
- **Paper**: [SLT 2022 - Language and Accent Recognition](https://arxiv.org/abs/2211.01922)

This model has been trained to recognize 12 common English accents from speech samples. It runs entirely **on a self-hosted Docker container**, ensuring no external API or third-party inference calls are made.

### 🗣️ Supported Accents
- African
- American
- Australian
- British
- Canadian
- Chinese
- Filipino
- Indian
- Irish
- New Zealand
- Scottish
- South African

---

### 🔄 Processing Pipeline

1. **Audio Input**  
   - Upload your own `.mp3` or `.wav` file, or provide a **YouTube or direct video/audio link**.
   - For video/audio links, the system extracts audio using `yt-dlp`.

2. **Conversion**  
   - All audio is normalized to **mono 16kHz WAV** using `pydub` and `FFmpeg`.

3. **Noise Reduction**  
   - The audio is cleaned using the `speechbrain/metricgan-plus-voicebank` model to reduce background noise and improve clarity.

4. **Accent Classification**  
   - The denoised WAV is passed to a **locally hosted** version of the `accent-id-commonaccent_xlsr-en-english` model for inference.

---

### ⚙️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Audio Processing**: `yt-dlp`, `pydub`, `torchaudio`, `FFmpeg`
- **Machine Learning**: SpeechBrain, PyTorch
- **Noise Reduction**: MetricGAN+ (VoiceBank)
- **Deployment**: Dockerized application hosted on [DigitalOcean](https://www.digitalocean.com/)
- **Model & Weights**: **Downloaded and stored inside the Docker container** (no external calls)

---

### 📦 Repositories & Resources
- [GitHub - JuanPZuluaga/accent-recog-slt2022](https://github.com/JuanPZuluaga/accent-recog-slt2022)
- [Hugging Face Model Card](https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-en-english)

""")

# Footer
st.markdown(
    """
    <hr>
    <div style='text-align: center; font-size: 0.9em'>
        Created by <strong>Mohammed Anas</strong><br>
        <a href='https://www.linkedin.com/in/mohammedanasa/' target='_blank'>LinkedIn</a> ·
        <a href='https://github.com/mohammedanasa/accent-detector-app' target='_blank'>GitHub Repo</a>
    </div>
    """,
    unsafe_allow_html=True
)
