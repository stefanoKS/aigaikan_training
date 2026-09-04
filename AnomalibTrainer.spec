# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = [("app/ui/assets/aigaikan_training.ico", "app/ui/assets")]
for package_name in ("matplotlib", "PySide6", "torchvision"):
    try:
        datas += collect_data_files(package_name, include_py_files=True)
    except Exception:
        pass

hiddenimports = collect_submodules("app")
for package_name in ("anomalib", "torch", "torchvision", "PySide6"):
    try:
        hiddenimports += collect_submodules(package_name)
    except Exception:
        pass

a = Analysis(
    ["app/main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AnomalibTrainer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon="app/ui/assets/aigaikan_training.ico",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AnomalibTrainer",
)
