# 🎙️ English Accent Classifier

This Streamlit app allows you to input a public video or audio URL (YouTube, MP3, MP4, etc.), extracts and denoises the audio, and then identifies the speaker's English accent using a pre-trained model from [SpeechBrain](https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-en-english).

## 🚀 Features

- ✅ Detects English accents such as Indian, British, American, and more.
- 🔊 Supports input via YouTube links, MP3 files, and test samples.
- 📈 Displays predicted accent with a confidence score.
- 🎛️ Audio denoising using [MetricGAN+](https://huggingface.co/speechbrain/metricgan-plus-voicebank) for improved accuracy.
- 🧪 Built-in test samples for easy evaluation.
- 🎧 Inline audio player for MP3/WAV files.

## 🧠 Models Used

- **Accent Detection**: [accent-id-commonaccent_xlsr-en-english](https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-en-english)  
  - Framework: [SpeechBrain](https://speechbrain.readthedocs.io)
  - Paper: [SLT 2022 - Language and Accent Recognition](https://arxiv.org/abs/2211.01922)

- **Denoising**: [metricgan-plus-voicebank](https://huggingface.co/speechbrain/metricgan-plus-voicebank)

## ⚙️ Tech Stack

- 🐍 Python
- 🎛️ Streamlit (UI)
- 🧠 SpeechBrain (ML models)
- 📦 Hugging Face (Model hosting)
- 🐳 Docker (for deployment)
- 🌐 Hosted on DigitalOcean
- 🔈 Pydub & ffmpeg for audio processing
- 🔊 yt-dlp for YouTube audio extraction

## 🧪 Demo

To run locally:

```bash
streamlit run app.py
