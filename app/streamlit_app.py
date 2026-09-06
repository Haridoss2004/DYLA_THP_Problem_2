"""Local Streamlit demonstration for the jewellery retrieval baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jewellery_retrieval.retrieval import JewelleryRetriever


INDEX_PATH = PROJECT_ROOT / "data" / "index" / "catalogue.faiss"
MANIFEST_PATH = PROJECT_ROOT / "data" / "index" / "catalogue_manifest.csv"
SUPPORTED_TYPES = ["jpg", "jpeg", "png", "webp"]


st.set_page_config(
    page_title="Dyla / Visual Retrieval",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink: #f4f1eb; --muted: #a8aaa5; --line: #2b302e; --panel: #151917; --accent: #d4a65a; --green: #9fcaaf; }
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
    .stApp { background: #0d100f; color: var(--ink); }
    .block-container { max-width: 1480px; padding: 2.3rem 3.2rem 4rem; }
    [data-testid="stSidebar"] { background: #111513; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] .block-container { padding: 2rem 1.3rem; }
    .brand { display:flex; align-items:center; gap:.7rem; margin-bottom:3.2rem; }
    .mark { color: var(--accent); font-size:1.45rem; line-height:1; }
    .brand-name { color:var(--ink); font-weight:800; letter-spacing:.16em; font-size:.78rem; }
    .eyebrow { color:var(--accent); font-family:'DM Mono', monospace; font-size:.68rem; letter-spacing:.16em; text-transform:uppercase; }
    h1 { font-size: clamp(2rem, 4vw, 4.6rem) !important; line-height:.98 !important; letter-spacing:-.04em !important; margin: .55rem 0 1rem !important; max-width: 780px; }
    .lede { color:var(--muted); font-size:1rem; line-height:1.7; max-width:640px; margin-bottom:2rem; }
    .section-rule { height:1px; background:var(--line); margin:1.6rem 0; }
    .metric { border-top:1px solid var(--line); padding:1rem 0 .3rem; }
    .metric-label { color:var(--muted); font-family:'DM Mono', monospace; font-size:.62rem; letter-spacing:.12em; text-transform:uppercase; }
    .metric-value { color:var(--ink); font-size:1.55rem; font-weight:700; margin-top:.28rem; }
    .upload-panel { background:var(--panel); border:1px solid var(--line); padding:1rem; }
    [data-testid="stFileUploader"] { background:#191e1b; border:1px dashed #485149; padding:.4rem; }
    [data-testid="stFileUploaderDropzone"] { min-height:150px; }
    .query-label { color:var(--muted); font-family:'DM Mono', monospace; font-size:.65rem; letter-spacing:.12em; text-transform:uppercase; margin:.3rem 0 .7rem; }
    .result-title { color:var(--ink); font-size:1.1rem; font-weight:700; margin:.2rem 0 1rem; }
    .result-card { background:var(--panel); border:1px solid var(--line); padding:.7rem; height:100%; }
    .result-rank { color:var(--accent); font-family:'DM Mono', monospace; font-size:.68rem; margin:.5rem 0 .35rem; }
    .result-id { color:var(--muted); font-family:'DM Mono', monospace; font-size:.63rem; margin-top:.6rem; }
    .result-score { color:var(--green); font-size:.95rem; font-weight:700; }
    .status { display:inline-flex; align-items:center; gap:.45rem; color:var(--green); font-family:'DM Mono', monospace; font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; }
    .status-dot { width:7px; height:7px; border-radius:50%; background:var(--green); display:inline-block; }
    .caption { color:var(--muted); font-size:.75rem; line-height:1.55; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the visual encoder and FAISS index...")
def load_retriever() -> JewelleryRetriever:
    if not INDEX_PATH.exists() or not MANIFEST_PATH.exists():
        raise FileNotFoundError("Retrieval index is missing. Run scripts/build_index.py first.")
    return JewelleryRetriever(INDEX_PATH, MANIFEST_PATH)


def metric(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


st.markdown('<div class="brand"><span class="mark">◆</span><span class="brand-name">DYLA / VISION LAB</span></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="eyebrow">Retrieval console</div>', unsafe_allow_html=True)
    st.markdown("### Catalogue search")
    st.markdown('<div class="caption">Image-level visual similarity over the local jewellery catalogue.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    top_k = st.slider("Candidates", min_value=3, max_value=5, value=5)
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">System</div>', unsafe_allow_html=True)
    metric("Catalogue images", "4,488")
    metric("Encoder", "ResNet-50")
    metric("Vector size", "2,048D")
    metric("Search", "FAISS / IP")
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="caption">Catalogue IDs are generated image-level identities. The source dataset does not provide verified product IDs.</div>', unsafe_allow_html=True)

st.markdown('<div class="eyebrow">Visual retrieval / 01</div>', unsafe_allow_html=True)
st.title("Find the closest jewellery images.")
st.markdown('<div class="lede">Upload a reference photograph and inspect the nearest catalogue candidates ranked by normalized visual similarity.</div>', unsafe_allow_html=True)

try:
    retriever = load_retriever()
except (FileNotFoundError, RuntimeError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

upload_column, query_column = st.columns([1, 1], gap="large")
with upload_column:
    st.markdown('<div class="query-label">01 / Input photograph</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop an image here", type=SUPPORTED_TYPES, label_visibility="collapsed")
    st.markdown('<div class="caption">JPG, JPEG, PNG, or WEBP. The image is processed locally by the selected encoder.</div>', unsafe_allow_html=True)

if uploaded is None:
    with query_column:
        st.markdown('<div class="query-label">02 / Query preview</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-panel"><div class="caption">Waiting for an image. The retrieval workspace will appear here after upload.</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="status"><span class="status-dot"></span>Ready for query</div>', unsafe_allow_html=True)
    st.stop()

try:
    query_image = Image.open(uploaded).convert("RGB")
except UnidentifiedImageError:
    st.error("Unsupported or corrupt image. Please upload a JPG, PNG, or WEBP file.")
    st.stop()

with query_column:
    st.markdown('<div class="query-label">02 / Query preview</div>', unsafe_allow_html=True)
    st.image(query_image, use_container_width=True)
    st.markdown(f'<div class="caption">{query_image.width} × {query_image.height}px · {uploaded.type}</div>', unsafe_allow_html=True)

with st.spinner("Comparing visual features..."):
    search_result = retriever.search(query_image, top_k=top_k)

st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
status_column, latency_column, signal_column = st.columns([1, 1, 2])
with status_column:
    st.markdown('<div class="status"><span class="status-dot"></span>Match candidates ready</div>', unsafe_allow_html=True)
with latency_column:
    metric("Search latency", f"{search_result['latency_ms']:.0f} ms")
with signal_column:
    st.markdown('<div class="metric"><div class="metric-label">Similarity meaning</div><div class="caption">Normalized embedding similarity. This is not a calibrated probability or product confidence.</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">03 / Ranked candidates</div>', unsafe_allow_html=True)
st.markdown('<div class="result-title">Nearest catalogue images</div>', unsafe_allow_html=True)

columns = st.columns(len(search_result["results"]), gap="small")
for column, result in zip(columns, search_result["results"]):
    with column:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        image_path = Path(result["image_path"])
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.warning("Catalogue image unavailable")
        st.markdown(f'<div class="result-rank">RANK {result["rank"]:02d}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-score">{result["similarity"]:.4f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="caption">{result["label"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-id">{result["catalogue_id"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)