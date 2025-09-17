# -*- mode: python ; coding: utf-8 -*-

# This PyInstaller spec file is configured for a professional build workflow.
# It centralizes all build configurations, ensuring reproducibility and clarity.
# The Analysis block discovers all necessary source files, modules, and libraries.
a = Analysis(
    ['integrated_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[('icon.ico', '.')],  # Correct way to include data files like icons
    # Explicitly include modules that PyInstaller's static analysis might miss.
    hiddenimports=[
        'openpyxl.cell._writer'  # Critical for Pandas Excel functionality.
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Exclude conflicting or unnecessary large packages.
    excludes=[
        'PyQt6'  # Resolves the PyQt5/PyQt6 binding conflict.
    ],
    noarchive=False,
    optimize=0,
)

# The PYZ block creates a compressed archive of all pure Python modules.
pyz = PYZ(a.pure)

# The EXE block defines the executable itself.
exe = EXE(
    pyz,
    a.scripts,
    [],
    name='TensorTool',  # Sets a clean name for the executable file.
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX is disabled for faster builds and better compatibility.
    runtime_tmpdir=None,
    console=False,  # This creates a windowed GUI application (no console).
    icon='icon.ico',  # Associates the icon with the executable.
)

# The COLLECT block gathers the EXE and all its dependencies into a single folder.
# This is the standard output for a one-directory build.
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TensorTool'  # This will be the name of the output folder in 'dist'.
)