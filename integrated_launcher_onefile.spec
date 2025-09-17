# -*- mode: python ; coding: utf-8 -*-

# Usage: pyinstaller integrated_launcher_onefile.spec
# This PyInstaller spec file is configured for a professional, ONE-FILE build.
a = Analysis(
    ['integrated_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[('icon.ico', '.')],
    hiddenimports=[
        'openpyxl.cell._writer'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt6'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# MODIFICATION: The EXE block now includes all dependencies (a.binaries, a.datas)
# that were previously in the COLLECT block. This is the key to a --onefile build.
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='TensorTool',  # Sets the name for the single executable file.
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Creates a windowed GUI application.
    icon='icon.ico',
)

# MODIFICATION: The COLLECT block is removed entirely for a --onefile build.
# coll = COLLECT(...)