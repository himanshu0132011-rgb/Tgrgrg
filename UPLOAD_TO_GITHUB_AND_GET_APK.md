# Get the BYTE Android APK using GitHub Actions

The repository includes `.github/workflows/build-android.yml`. Once the project contents are pushed to GitHub, GitHub Actions builds a debug APK in the cloud and uploads it as an artifact.

## Push the project from Android using Termux

1. Install Termux from its official F-Droid or GitHub source.
2. Download `BYTE_v0.2_Android_Project.zip` to your phone's Downloads folder.
3. In Termux, run:

```bash
pkg update -y
pkg install git unzip -y
termux-setup-storage
mkdir -p ~/byte-upload
cd ~/byte-upload
unzip ~/storage/downloads/BYTE_v0.2_Android_Project.zip
cd BYTE
git init
git branch -M main
git add .
git config --global user.name "Your GitHub Name"
git config --global user.email "YOUR_EMAIL@example.com"
git commit -m "Add BYTE v0.2 Android project"
```

4. Create an empty repository on GitHub, for example `BYTE`.
5. Connect and push (replace the placeholder with your repository URL):

```bash
git remote add origin https://github.com/YOUR_USERNAME/BYTE.git
git push -u origin main
```

GitHub HTTPS pushes require authentication. Use a GitHub personal access token with permission to write repository contents when Git asks for a password. Do not share the token with anyone or put it into the source files.

## Download and install the APK

1. Open your repository on GitHub.
2. Open the **Actions** tab.
3. Open the latest **Build BYTE Android APK** workflow run.
4. Under **Artifacts**, download `BYTE-debug-apk`.
5. Extract the downloaded artifact ZIP. It contains `app-debug.apk`.
6. Tap `app-debug.apk` and approve Android's install-from-this-source prompt if it appears.

The workflow creates a debug APK, which is suitable for personal testing. It is not a Play Store release build and is not signed with a private release key.

## If the workflow does not start

Check that `.github/workflows/build-android.yml` exists in the repository at exactly that path and that the files were pushed to the `main` branch. You can also start it manually from the Actions tab using **Run workflow**.
