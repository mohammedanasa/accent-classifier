# 🎙️ English Accent Classifier

This Streamlit app allows you to input a public video URL (MP4, Loom, etc.), extracts the audio, and identifies the speaker's English accent (e.g., US, British, Indian, etc.) using a pre-trained model from [SpeechBrain](https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-en-english).

## 🚀 Features
- Detects English accent from spoken video.
- Shows predicted accent and confidence score.
- Works with most MP4 or Loom links.

## 🧠 Model
[Accent ID Model by Juan Zuluaga](https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-en-english)

## 🧪 Demo
Deploy or run locally using:

```bash
streamlit run app.py
```

## 📦 Requirements
Install with:

```bash
pip install -r requirements.txt
```

Make sure you have `ffmpeg` installed on your system.
