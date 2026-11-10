# BYTE v0.2 — local mini-model + offline agent

BYTE is a deliberately small educational AI experiment. It now has a desktop Python version and a native Android project.

## What is included

- Custom corpus-built word tokenizer, vocabulary, and token IDs.
- A tiny neural next-token model with randomly initialized weights, trained using backpropagation and Adam.
- A training pipeline and local inference/generation.
- An offline agent router with local note save/search/recall, basic arithmetic, help, and status tools.
- A native Android app project that runs the bundled weights on-device with pure Java inference.

**Reality check:** BYTE is not an LLM, transformer, or ChatGPT replacement. Its small feed-forward model learns short text patterns from a tiny corpus. Its agent tools are explicitly routed and limited; general messages are not reliably answered. There is no browser, cloud API, or internet access in the Android app.

## Desktop

Requirements: Python 3.10+ and NumPy.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python byte.py status
python byte.py chat
```

Agent commands: `/help`, `status`, `remember <note>`, `recall`, `search memory for <words>`, `calculate <expression>`, and `generate <prompt>`.

To retrain from random weights using your corpus and memory notes:

```bash
python byte.py train
python tools/export_android_model.py  # refresh Android model assets after retraining
```

## Android

The `android/` folder is a native Android Studio project. This environment does not contain Android SDK/Gradle, so no compiled APK is included. The repository includes a GitHub Actions workflow that builds a debug APK in the cloud on push. Follow [`UPLOAD_TO_GITHUB_AND_GET_APK.md`](UPLOAD_TO_GITHUB_AND_GET_APK.md) to push the files and download the APK from the Actions artifact.

## Privacy and security

- Desktop app writes memory under `data/memory.txt`.
- Android app writes memory to its private app storage.
- Android manifest requests no network permission.
- Only run source code and install packages you trust.
