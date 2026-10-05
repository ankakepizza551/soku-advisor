# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['soku_advisor_app.py'],
    pathex=[],
    binaries=[],
    datas=[('analyzer.py', '.'), ('soku_live_reader.py', '.'), ('char_advisor.py', '.'), ('ai_advisor.py', '.'), ('char_data.json', '.'), ('chart.umd.min.js', '.'), ('soku_advisor.ico', '.')],
    hiddenimports=['cv2', 'numpy', 'tkinter', 'player_history'],
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
    a.binaries,
    a.datas,
    [],
    name='SokuAdvisor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['soku_advisor.ico'],
    manifest='app.manifest',
)
