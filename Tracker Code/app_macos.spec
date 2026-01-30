# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for macOS Screenshot Tracker
# Build with: pyinstaller app_macos.spec

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all data files and hidden imports
datas = []
hiddenimports = [
    'pytz',
    'pytz.tzfile',
    'pytz.lazy',
    'mss',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'requests',
    'urllib3',
    'psutil',
    'pickle',
    'json',
    'csv',
    'io',
    'threading',
    'queue',
    'collections',
    'dataclasses',
    'datetime',
    'socket',
    'ssl',
    'tempfile',
    'ipaddress',
]

# Add any additional data files if needed (configs, assets, etc.)
# Example: datas += [('configs', 'configs')]

a = Analysis(
    ['Omama_Anwar_14_tracker.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ScreenshotTracker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False for GUI apps (no console window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

app = BUNDLE(
    exe,
    name='ScreenshotTracker.app',
    icon=None,  # Add path to .icns file if you have an app icon
    bundle_identifier='com.tracker.screenshottracker',
    version='1.0.0',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSHighResolutionCapable': 'True',
        'NSSupportsAutomaticGraphicsSwitching': 'True',
        'LSMinimumSystemVersion': '10.13',  # macOS High Sierra minimum
        'NSRequiresAquaSystemAppearance': 'False',
        # Screen Recording permission description (shown in System Preferences)
        # Note: Screen Recording permission is handled by macOS automatically when using mss
        # No Info.plist key is required for screen recording, but we include these for completeness
        'NSMicrophoneUsageDescription': 'Not used by this application.',
    },
)
