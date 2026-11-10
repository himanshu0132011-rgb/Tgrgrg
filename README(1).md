# BYTE Android v0.2

A native Android shell for the BYTE from-scratch mini language-model experiment. The app runs the bundled checkpoint directly on the phone and adds a small offline tool-using agent layer.

## What the Android app can do

- Run BYTE's bundled tiny MLP locally using a pure-Java inference port.
- Use BYTE's own corpus-built word vocabulary and token IDs.
- Save notes to the app's private storage (`remember <note>`).
- List notes (`recall`) and search notes (`search memory for <words>`).
- Calculate basic arithmetic (`calculate 18 * (4 + 2)`).
- Show local model status (`status`) and command help (`/help`).
- Generate a raw text continuation (`generate <prompt>`).

The app requests **no internet permission**. Notes stay in the app's private storage. The bundled neural model is only a tiny next-token predictor, not a capable LLM. The deterministic agent router handles the explicit local tools; it does not have broad reasoning, browser access, or autonomous internet tools.

## Build and download the APK (recommended)

This workspace does not contain the Android SDK or Gradle, so it cannot compile an APK locally. The repository includes a GitHub Actions workflow that does the build in GitHub's cloud. You do not need Android Studio or an Android SDK on your phone.

1. Push the complete project to a GitHub repository, including the hidden `.github/workflows/build-android.yml` file.
2. Open the repository's **Actions** tab.
3. Open the latest **Build BYTE Android APK** run.
4. Download the `BYTE-debug-apk` artifact.
5. Extract the artifact ZIP, then tap `app-debug.apk` on the phone. Approve Android's install-from-this-source prompt if requested.

See [`../UPLOAD_TO_GITHUB_AND_GET_APK.md`](../UPLOAD_TO_GITHUB_AND_GET_APK.md) for the full Termux/GitHub instructions.

The workflow uses JDK 17, Gradle 8.9, Android Gradle Plugin 8.7.3, and Android SDK Platform 35. It produces a debug APK for personal testing, not a Play Store release. For a release APK, build and sign it with your own private release key; never share that key or its passwords.

## App details

- Application ID: `com.byteai.local`
- Version: `0.2.0` (version code 2)
- Minimum Android: Android 6.0 (API 23)
- Target SDK: 35
- No third-party runtime libraries
- No API keys, cloud calls, or internet permission

## Model bundle format

`app/src/main/assets/byte_weights.bin` contains the current trained checkpoint converted from the Python NumPy `.npz` into a simple big-endian float format. `tokenizer.json` contains the same vocabulary. `ByteEngine.java` performs the model's embedding lookup, tanh hidden layer, softmax-equivalent top-k sampling, and token decoding. This is inference only; on-device training and editing the vocabulary are not implemented in this APK.

## Test commands (desktop Python version)

From the parent `BYTE` folder:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python byte.py chat
```

Useful agent commands:

```text
/help
status
remember BYTE is my local AI project
recall
search memory for BYTE
calculate 18 * (4 + 2)
generate BYTE is
```
