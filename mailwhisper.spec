# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_dynamic_libs

binaries = []
binaries += collect_dynamic_libs('ctranslate2')
binaries += collect_dynamic_libs('tokenizers')
binaries += collect_dynamic_libs('onnxruntime')
binaries += collect_dynamic_libs('sounddevice')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=[('ui/icons', 'ui/icons'), ('img', 'img')],
    hiddenimports=['AVFoundation'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MailWhisper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    icon=['img/app.icns'],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MailWhisper',
)

INFO_PLIST = {
    "CFBundleName": "MailWhisper",
    "CFBundleDisplayName": "MailWhisper",
    "CFBundleIdentifier": "com.mailwhisper.app",
    "CFBundleShortVersionString": "1.0.0",
    "CFBundleVersion": "1",
    "LSMinimumSystemVersion": "10.13",
    "LSApplicationCategoryType": "public.app-category.productivity",
    "NSHighResolutionCapable": True,
    "NSMicrophoneUsageDescription": "Used to record your voice for transcription.",
}

app = BUNDLE(
    coll,
    name='MailWhisper.app',
    icon='img/app.icns',
    bundle_identifier='com.mailwhisper.app',
    info_plist=INFO_PLIST,
)