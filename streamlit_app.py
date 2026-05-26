"""
Streamlit Cloud entry point — Company Enrichment App.

DEPLOYMENT INSTRUCTIONS
-----------------------
In Streamlit Cloud → your app → Settings → General:
  Main file path → streamlit_app.py

commercial_fit_scoring.py is a helper/scoring module only.
enrich_clients_claude.py contains the full application logic.
This file is the thin Streamlit launcher that Cloud executes.

If you cannot find "Main file path" in the UI you must delete the existing
deployment and redeploy from this branch, choosing streamlit_app.py as the
main file during the "Deploy an app" wizard.
"""

import os
import pathlib
import base64

os.environ.setdefault("_STREAMLIT_ENTRYPOINT", "1")

import streamlit as st

st.set_page_config(
    page_title="mYngle · lead prioritizer",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _ensure_padded_logo() -> pathlib.Path:
    """
    Return path to the best available mYngle logo asset, in preference order:
      1. Mynglelogofinal_fixed.png  — reconstructed circle top (best)
      2. Mynglelogofinal_padded.png — padded but dot still cropped
      3. Mynglelogofinal.jpg        — original (dot clipped at top)
    Creates Mynglelogofinal_padded.png on-the-fly if neither fixed nor padded exist.
    """
    base  = pathlib.Path(__file__).parent
    fixed = base / "Mynglelogofinal_fixed.png"
    if fixed.exists():
        return fixed

    src = base / "Mynglelogofinal.jpg"
    dst = base / "Mynglelogofinal_padded.png"

    if dst.exists():
        return dst
    if not src.exists():
        return src

    from PIL import Image
    img = Image.open(src).convert("RGBA")
    pad_top, pad_bottom, pad_left, pad_right = 70, 30, 25, 25
    canvas = Image.new(
        "RGBA",
        (img.width + pad_left + pad_right, img.height + pad_top + pad_bottom),
        (255, 255, 255, 255),
    )
    canvas.paste(img, (pad_left, pad_top))
    canvas.save(dst)
    return dst


_logo_path = _ensure_padded_logo()
_mime      = "image/png" if _logo_path.suffix.lower() == ".png" else "image/jpeg"
_logo_src  = (
    f"data:{_mime};base64,{base64.b64encode(_logo_path.read_bytes()).decode()}"
    if _logo_path.exists() else ""
)
_img_tag = (
    f'<img src="{_logo_src}" class="brand-logo" alt="mYngle" />'
    if _logo_src else ""
)

# Debug: logo verification — remove once confirmed correct
if _logo_path.exists():
    from PIL import Image as _PIL_dbg
    import numpy as _np_dbg
    _dbg_img = _PIL_dbg.open(_logo_path).convert("RGBA")
    _dbg_arr = _np_dbg.array(_dbg_img)
    _dbg_w, _dbg_h = _dbg_img.size
    # Find first row with an orange pixel (R>150, G<100, B<50, A>200)
    _dbg_top_y = None
    for _dbg_y in range(_dbg_h):
        row = _dbg_arr[_dbg_y]
        if _np_dbg.any((row[:,0]>150) & (row[:,1]<100) & (row[:,2]<50) & (row[:,3]>200)):
            _dbg_top_y = _dbg_y
            break
    st.caption(f"Logo file: {_logo_path.name} | size: {_dbg_w}×{_dbg_h} | top orange y: {_dbg_top_y}")

st.markdown(
    f"""
    <style>
    .block-container {{
        max-width: 880px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }}

    div[data-testid="stMarkdownContainer"]:has(.brand-header) {{
        overflow: visible !important;
        margin-bottom: 1.0rem;
    }}

    .brand-header {{
        display: grid;
        grid-template-columns: 43% 57%;
        align-items: center;
        min-height: 140px;
        padding-top: 10px;
        padding-bottom: 6px;
        overflow: visible !important;
    }}

    .brand-title-block {{
        display: flex;
        align-items: center;
        justify-content: flex-start;
        overflow: visible !important;
    }}

    .brand-title {{
        font-size: 42px;
        font-weight: 700;
        color: #0B1F3A;
        line-height: 1.1;
        white-space: nowrap;
        margin: 0;
        padding: 0;
    }}

    .brand-logo-block {{
        display: flex;
        justify-content: flex-end;
        align-items: center;
        padding: 0;
        line-height: 0;
        overflow: visible !important;
    }}

    .brand-logo {{
        width: 430px;
        max-width: 100%;
        height: auto;
        display: block;
        object-fit: contain;
        object-position: center center;
        overflow: visible !important;
    }}
    </style>

    <div class="brand-header">
      <div class="brand-title-block">
        <span class="brand-title">lead prioritizer</span>
      </div>
      <div class="brand-logo-block">
        {_img_tag}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

import runpy
_MAIN = pathlib.Path(__file__).parent / "enrich_clients_claude.py"
runpy.run_path(str(_MAIN), run_name="__main__")
