# Offline Multilingual Doctor–Patient Speech-to-Text Pipeline

A complete offline-first, multilingual (Marathi, Hindi, English, and code-mixed) Speech-to-Text (STT) pipeline tailored for clinical hospital conversations, symptom tracking, medicine prescription analysis, and Doctor-Patient dialogue.

---

## 🏛️ Architecture & Division of Responsibility

```
Doctor-Patient Audio (.wav, .mp3, .m4a, .flac)
                    ↓
[1] Audio Preprocessing (16kHz Mono WAV Peak-Normalized)
                    ↓
[2] Sarvam AI STT API (saaras:v2 - Multilingual / Code-Mixed)
                    ↓
[3] Clinical / Medical Text Normalization (Medicines, BP, HbA1c, Units)
                    ↓
[4] Human-in-the-Loop Review Studio (Web UI & REST API)
                    ↓
[5] Dataset Builder (80% Train, 10% Validation, 10% Test Manifests)
                    ↓
[6] Local Offline Fine-Tuning Engine (CUDA / AMP / Checkpointing / CTC)
                    ↓
[7] Accuracy Evaluation (WER, CER, Medical Terminology Retention)
                    ↓
[8] Offline Model Checkpoint (best_model.pt) & Offline Inference
```

### 🔍 Important Architectural Design (Sarvam vs Local Fine-Tuning):
- **Sarvam AI Role:** Used for high-accuracy initial automatic transcription and multilingual transcription generation (`saaras:v2` / Sarvam-M). Sarvam does not provide a public weights fine-tuning endpoint.
- **Local Trainable Model Role:** Used for 100% offline, local custom model fine-tuning with NVIDIA GPU / CUDA support, PyTorch AMP (Automatic Mixed Precision), CTC acoustic model architecture, checkpoint saving/resuming, and validation WER/CER tracking.

---

## 📂 Project Structure

```
backend/
├── app.py                         # FastAPI backend integrating API routes & Review Studio UI
├── config.yaml                    # Master configuration (audio specs, normalizer rules, training params)
├── .env.example                   # Environment variable template (SARVAM_API_KEY)
├── requirements.txt               # All pipeline dependencies
├── audio_pipeline/
│   ├── preprocess.py              # Recursive scan, 16kHz mono conversion, safe normalization, error logging
│   ├── transcribe.py              # Sarvam AI STT client with code-mixed preservation
│   ├── normalizer.py              # Medical terminology, medicines, numbers, units, abbreviations
│   ├── dataset_builder.py         # 80/10/10 split generator & NeMo JSONL / CSV manifests
│   └── evaluate.py                # WER, CER, and clinical accuracy evaluation
├── training/
│   ├── dataset.py                 # PyTorch ASR Dataset, Devanagari/English vocabulary & batch collator
│   ├── train.py                   # Local offline trainer with GPU auto-detection, AMP, checkpoints
│   └── infer.py                   # Standalone offline inference on trained best_model.pt
├── datasets/
│   ├── raw_audio/                 # Place raw input audio files here (.wav, .mp3, .m4a, .flac)
│   ├── processed_audio/           # Standardized 16kHz mono WAV files
│   ├── transcripts/               # Raw and normalized transcription files (transcripts_master.json)
│   └── manifests/                 # train.jsonl, validation.jsonl, test.jsonl, dataset.csv
├── outputs/
│   ├── checkpoints/               # Saved model checkpoints & best_model.pt
│   ├── logs/                      # Training logs & pipeline error logs
│   └── reports/                   # WER, CER, and clinical reports (evaluation_report.json)
└── README.md                      # Complete documentation
```

---

## 🚀 Quickstart Guide

### 1. Configure Environment
Copy `.env.example` to `.env` and configure your Sarvam API Key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
SARVAM_API_KEY=your_actual_sarvam_key_here
PORT=8000
HOST=0.0.0.0
```

### 2. Start the Interactive Studio & API Server
```bash
cd backend
python app.py
```
Open your browser at `http://localhost:8000` to access the **Clinical Review Studio & Pipeline Dashboard**.

---

## 🛠️ CLI Pipeline Execution

You can run each stage independently via command-line:

### Step 1: Preprocess Audio
Recursively finds all supported audio (`.wav`, `.mp3`, `.m4a`, `.flac`) in `datasets/raw_audio/`, converts them to 16kHz mono WAV, normalizes peaks safely, and logs corrupted files:
```bash
python audio_pipeline/preprocess.py --raw_dir datasets/raw_audio --output_dir datasets/processed_audio
```

### Step 2: Sarvam AI Multilingual Transcription
Transcribes 16kHz audio using Sarvam AI, preserving multilingual mixed speech (Marathi, Hindi, English):
```bash
python audio_pipeline/transcribe.py --processed_dir datasets/processed_audio
```

### Step 3: Medical Text Normalization & Dataset Splitting
Standardizes medicines (Metformin, Insulin, Paracetamol, etc.), clinical abbreviations (BP, PR, SpO2, HbA1c), numbers, dates, and builds 80/10/10 train/validation/test manifests:
```bash
python audio_pipeline/dataset_builder.py --transcripts datasets/transcripts/transcripts_master.json --include_pending
```

### Step 4: Offline Local Model Fine-Tuning
Trains acoustic model locally with automatic GPU/CUDA detection, AMP mixed precision, checkpointing, and validation WER tracking:
```bash
python training/train.py --config config.yaml --epochs 15
```
To resume from a saved checkpoint:
```bash
python training/train.py --resume outputs/checkpoints/checkpoint_epoch_5.pt
```

### Step 5: Evaluate Model Accuracy
Evaluates trained model against the test split, calculating Word Error Rate (WER), Character Error Rate (CER), and medical term retention:
```bash
python audio_pipeline/evaluate.py
```

### Step 6: Test Offline Inference
Run offline speech recognition on any new audio file using `best_model.pt`:
```bash
python training/infer.py --audio datasets/processed_audio/sample1_16k_mono.wav --checkpoint outputs/checkpoints/best_model.pt
```

---

## 🩺 Supported Clinical Normalizations
- **Diabetes:** Metformin, Glimepiride, Vildagliptin, Sitagliptin, Dapagliflozin, Empagliflozin, Insulin, RBS, FBS, HbA1c
- **Cardiovascular:** Telmisartan, Amlodipine, Atenolol, Atorvastatin, Rosuvastatin, Ecosprin, BP (रक्तदाब), ECG
- **General / Antibiotics:** Paracetamol, Dolo 650, Pantoprazole, Pan-D, Amoxicillin, Augmentin, Azithromycin, Cetirizine
- **Dosages & Timing:** OD, BD, TDS, SOS, Tab, Cap, Inj, Syp, mg, ml, mmHg
