# Screenshot Tracker - macOS Build Guide

This application captures screenshots and uploads them to a remote server. This guide covers building and running the application on macOS.

## Prerequisites

- **macOS 10.13 (High Sierra) or later**
- **Python 3.10 or later**
- **Internet connection** (for installing dependencies and uploading screenshots)

## Quick Start

### 1. Install Dependencies

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run in Development Mode

```bash
python Omama_Anwar_14_tracker.py
```

### 3. Build macOS Application Bundle

```bash
# Make build script executable (first time only)
chmod +x build_macos.sh

# Run build script
./build_macos.sh
```

The build script will:
- Create a virtual environment
- Install all dependencies
- Build the `.app` bundle using PyInstaller
- Output: `dist/ScreenshotTracker.app`

### 4. Run the Built Application

```bash
open dist/ScreenshotTracker.app
```

Or double-click `ScreenshotTracker.app` in Finder.

## Getting the macOS Build from GitHub (No Mac Required)

The repository includes a **GitHub Actions workflow** that builds `ScreenshotTracker.app` on a macOS runner. You can get a ready-to-use `.app` without having a Mac:

1. Push the repo to GitHub (with the `.github/workflows/build-macos-app.yml` workflow and the `Github Mac Build` / `Tracker Code` folders).
2. Go to **Actions** → **Build macOS App**.
3. Run the workflow: click **Run workflow** (or it runs automatically on push to `main`/`master` when the tracker script or build files change).
4. When the run finishes, open the run and download the **ScreenshotTracker-macOS** artifact (a zip containing `ScreenshotTracker.app`).
5. Unzip on a Mac and move `ScreenshotTracker.app` to Applications or any folder. On first launch, see **macOS Gatekeeper & Permissions** below.

## macOS Gatekeeper & Permissions

### First Run - Gatekeeper Warning

On first launch, macOS Gatekeeper may block the application because it's not code-signed. You'll see a message like:

> "ScreenshotTracker.app" cannot be opened because the developer cannot be verified.

#### Solution: Remove Quarantine Attribute

Run this command in Terminal to allow the app to run:

```bash
xattr -rd com.apple.quarantine dist/ScreenshotTracker.app
```

Then try opening the app again.

**Note:** This is safe for local builds. For distribution, you should code-sign the app with an Apple Developer certificate.

### Screen Recording Permission

The app requires **Screen Recording** permission to capture screenshots.

1. On first launch, macOS will prompt you to grant permission
2. Go to **System Preferences** → **Security & Privacy** → **Privacy** → **Screen Recording**
3. Check the box next to `ScreenshotTracker.app`
4. Restart the app if needed

**Important:** Without this permission, screenshot capture will fail silently.

## Configuration

### Server Configuration

Edit `Omama_Anwar_14_tracker.py` to configure:

- `SERVER_BASE`: Upload server URL (line 42)
- `BACKEND_API_BASE`: Backend API URL (line 51)
- `SCREENSHOTS_PER_SECOND`: Capture rate (line 44)

### Writer ID from Filename

The application extracts the Writer ID from the script filename:
- Format: `{WriterName}_{ID}_tracker.py`
- Example: `Omama_Anwar_14_tracker.py` → Writer: "Omama Anwar", ID: 14

## Building for Distribution

### Adding an App Icon

1. Create an `.icns` file (use `iconutil` or online converters)
2. Update `app_macos.spec`:
   ```python
   icon='path/to/icon.icns',
   ```

### Code Signing (Optional, for Distribution)

For distribution outside your Mac, code-sign the app:

```bash
codesign --deep --force --verify --verbose --sign "Developer ID Application: Your Name" dist/ScreenshotTracker.app
```

Replace `"Developer ID Application: Your Name"` with your actual certificate name.

### Notarization (Optional, for Distribution)

For macOS 10.14.5+, notarize the app for distribution:

```bash
# Create a zip for notarization
ditto -c -k --keepParent dist/ScreenshotTracker.app ScreenshotTracker.zip

# Submit for notarization (requires Apple Developer account)
xcrun altool --notarize-app --primary-bundle-id "com.tracker.screenshottracker" \
  --username "your@email.com" --password "@keychain:AC_PASSWORD" \
  --file ScreenshotTracker.zip
```

## Troubleshooting

### App Crashes on Launch

1. **Check Console.app** for crash logs:
   ```bash
   open /Applications/Utilities/Console.app
   ```
   Look for `ScreenshotTracker` entries.

2. **Check permissions**: Ensure Screen Recording permission is granted

3. **Check logs**: The app creates `app.log` in the same directory

### Screenshots Not Capturing

1. **Verify Screen Recording permission** (see above)
2. **Check network connectivity** to upload server
3. **Check `app.log`** for error messages

### Build Fails

1. **Ensure Python 3.10+**: `python3 --version`
2. **Clean build artifacts**: `rm -rf build/ dist/ __pycache__/`
3. **Reinstall dependencies**: `pip install --upgrade -r requirements.txt`
4. **Check PyInstaller version**: `pip install --upgrade pyinstaller`

### Import Errors in Built App

If you see import errors, add missing modules to `hiddenimports` in `app_macos.spec`:

```python
hiddenimports = [
    'your_missing_module',
    # ... existing imports
]
```

## Development Notes

### Cross-Platform Compatibility

This application is designed to work on:
- **macOS** (primary target)
- **Windows** (legacy support)
- **Linux** (requires X11 display)

### File Paths

All file paths use `os.path.join()` for cross-platform compatibility. The `resource_path()` helper ensures PyInstaller bundles work correctly.

### System Information Collection

The app collects system metadata using platform-specific commands:
- **macOS**: `sysctl`, `system_profiler`
- **Windows**: `wmic` (legacy)
- **Linux**: `lscpu`, `lspci`

## Project Structure

```
.
├── Omama_Anwar_14_tracker.py  # Main application
├── requirements.txt            # Python dependencies
├── app_macos.spec             # PyInstaller spec file
├── build_macos.sh             # Build script
├── README.md                  # This file
└── dist/                      # Build output (created by build script)
    └── ScreenshotTracker.app  # macOS application bundle
```

## License

[Add your license information here]

## Support

For issues or questions, check:
1. `app.log` for detailed error messages
2. Console.app for system-level errors
3. Network connectivity to upload server
