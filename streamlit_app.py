# streamlit_app.py
# -------------------------------------------------------------
# Pinterest Board Color & Trend Analyzer — URL-only, board-only
# -------------------------------------------------------------
# - Paste a PUBLIC Pinterest board URL and click Analyze.
# - Only analyzes pins that BELONG to that board (excludes "More like this").
# - If Pinterest's embedded JSON isn't accessible, falls back to board RSS.
# - Color-true charts (bars/dots use real hex).
# - Visuals: Master palette, Pin Gallery, HSV insights, Color Waffle,
#            Per-cluster boxplots, Hue×Value heatmap, Dominant Color Sequence.
# - Diagnostics expander shows which method (JSON/RSS) was used.
# -------------------------------------------------------------

import io
import re
import math
import json
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import requests
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans

import streamlit as st
import altair as alt


# ===============================
# Utilities
# ===============================

def hex_from_rgb(rgb_tuple):
    r, g, b = [int(max(0, min(255, round(c)))) for c in rgb_tuple]
    return f"#{r:02x}{g:02x}{b:02x}"

def rgb_to_hsv_np(rgb):
    """RGB [0-255] -> HSV [0-1], vectorized."""
    rgb = np.asarray(rgb, dtype=np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    diff = mx - mn
    h = np.zeros_like(mx)
    mask = diff != 0
    r_eq = (mx == r) & mask
    g_eq = (mx == g) & mask
    b_eq = (mx == b) & mask
    h[r_eq] = ((g[r_eq] - b[r_eq]) / diff[r_eq]) % 6
    h[g_eq] = ((b[g_eq] - r[g_eq]) / diff[g_eq]) + 2
    h[b_eq] = ((r[b_eq] - g[b_eq]) / diff[b_eq]) + 4
    h = (h / 6.0) % 1.0
    s = np.where(mx == 0, 0, diff / mx)
    v = mx
    return np.stack([h, s, v], axis=-1)

@st.cache_data(show_spinner=False)
def fetch_image_bytes(url, timeout=15):
    try:
        r = requests.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
        )
        r.raise_for_status()
        return r.content
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_image_as_array(url, max_side=512):
    data = fetch_image_bytes(url)
    if not data:
        return None
    try:
        with Image.open(io.BytesIO(data)) as img:
            img = img.convert("RGB")
            w, h = img.size
            scale = max(w, h) / max_side if max(w, h) > max_side else 1
            if scale > 1:
                img = img.resize((int(w/scale), int(h/scale)), Image.LANCZOS)
            return np.array(img)
    except Exception:
        return None


# ===============================
# Scraper (board-only)
# ===============================

PINTEREST_IMG_RE = re.compile(r'https?://i\.pinimg\.com/[^\s">]+\.(?:jpg|jpeg|png|webp)')

def _normalize_board_path(u: str):
    try:
        p = urlparse(u)
        parts = [x for x in (p.path or "").lower().strip("/").split("/") if x]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return None
    except Exception:
        return None

def _obj_board_url(o: dict):
    if not isinstance(o, dict):
        return None
    b = o.get("board")
    if isinstance(b, dict):
        for k in ("url", "board_url"):
            if isinstance(b.get(k), str):
                return b.get(k)
    for k in ("board_url", "boardUrl", "grid_board_url", "gridBoardUrl"):
        if isinstance(o.get(k), str):
            return o.get(k)
    return None

def _find_target_board_ids(data, target_path: str):
    ids = set()
    def walk(o):
        if isinstance(o, dict):
            url = o.get("url") or o.get("board_url")
            if isinstance(url, str):
                if target_path and target_path == (_normalize_board_path(url) or ""):
                    for k in ("id", "board_id"):
                        v = o.get(k)
                        if isinstance(v, (str, int)):
                            ids.add(str(v))
            b = o.get("board")
            if isinstance(b, dict):
                burl = b.get("url") or b.get("board_url")
                if isinstance(burl, str) and target_path == (_normalize_board_path(burl) or ""):
                    v = b.get("id")
                    if isinstance(v, (str, int)):
                        ids.add(str(v))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(data)
    return ids

@st.cache_data(show_spinner=False)
def extract_pins_from_html(html: str, board_url: str, max_pins: int = 200):
    """
    Extract ONLY pins that belong to the given board URL (exclude recommendations).
    Uses embedded JSON (__PWS_DATA__) and filters pins by board url/id.
    """
    target_path = _normalize_board_path(board_url)
    m = re.search(
        r'<script[^>]+id="__PWS_DATA__"[^>]*>(.*?)</script>',
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return []  # can't safely separate without JSON
    try:
        data = json.loads(m.group(1))
        target_ids = _find_target_board_ids(data, target_path)
        pins = []

        def walk(o):
            if isinstance(o, dict):
                # detect image
                images = o.get("images") or (o.get("image") if isinstance(o.get("image"), dict) else None)
                img_url = None
                if isinstance(images, dict):
                    for key in ("orig", "736x", "474x", "170x", "small", "medium", "large"):
                        if key in images and isinstance(images[key], dict) and images[key].get("url"):
                            img_url = images[key]["url"]
                            break

                # decide if belongs to target board
                belongs = False
                burl = _obj_board_url(o)
                if isinstance(burl, str) and target_path:
                    belongs = (target_path == (_normalize_board_path(burl) or ""))
                if not belongs:
                    bid = o.get("board_id")
                    if bid is None and isinstance(o.get("board"), dict):
                        bid = o["board"].get("id")
                    if bid is not None and str(bid) in target_ids:
                        belongs = True

                if img_url and belongs:
                    pin_id = o.get("id") or o.get("pin_id")
                    title = o.get("title") or o.get("grid_title") or o.get("alt_text") or ""
                    description = o.get("description") or o.get("grid_description") or ""
                    created_at = o.get("created_at") or o.get("created") or None
                    pins.append(
                        {
                            "pin_id": str(pin_id) if pin_id else None,
                            "title": title,
                            "description": description,
                            "created_at": created_at,
                            "image_url": img_url,
                        }
                    )
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)

        walk(data)

        # dedup & limit
        dedup, seen = [], set()
        for p in pins:
            url = p.get("image_url")
            if url and url not in seen:
                dedup.append(p)
                seen.add(url)
            if len(dedup) >= max_pins:
                break
        return dedup
    except Exception:
        return []

@st.cache_data(show_spinner=True)
def fetch_board_rss(board_url: str, max_items: int = 200):
    """
    Fallback: fetch the board's RSS feed and extract image URLs from official pins only.
    Example: https://www.pinterest.com/<username>/<board-slug>.rss
    """
    try:
        p = urlparse(board_url)
        parts = [x for x in (p.path or "").strip("/").split("/") if x]
        if len(parts) < 2:
            return []
        rss_url = f"{p.scheme}://{p.netloc}/{parts[0]}/{parts[1]}.rss"
        r = requests.get(
            rss_url,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
            },
        )
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.text)
        ns = {"media": "http://search.yahoo.com/mrss/"}
        items = root.findall(".//item")
        pins = []
        for it in items[:max_items]:
            title_el = it.find("title")
            title = title_el.text if title_el is not None else ""
            pub_el = it.find("pubDate")
            created_at = pub_el.text if pub_el is not None else None

            img_url = None
            mcontent = it.find("media:content", ns)
            if mcontent is not None:
                img_url = mcontent.attrib.get("url")
            if not img_url:
                mthumb = it.find("media:thumbnail", ns)
                if mthumb is not None:
                    img_url = mthumb.attrib.get("url")
            if not img_url:
                encl = it.find("enclosure")
                if encl is not None:
                    img_url = encl.attrib.get("url")
            if not img_url:
                desc_el = it.find("description")
                desc = desc_el.text if desc_el is not None else ""
                m = PINTEREST_IMG_RE.search(desc or "")
                if m:
                    img_url = m.group(0)

            if img_url:
                pins.append(
                    {
                        "pin_id": None,
                        "title": title,
                        "description": "",
                        "created_at": created_at,
                        "image_url": img_url,
                    }
                )
        return pins
    except Exception:
        return []

@st.cache_data(show_spinner=True)
def fetch_board_html(board_url: str) -> str:
    r = requests.get(
        board_url,
        timeout=20,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    r.raise_for_status()
    return r.text

@st.cache_data(show_spinner=True)
def scrape_board_boardonly(board_url: str, max_pins: int = 200):
    """
    Strict board-only scrape:
    1) Try embedded JSON (__PWS_DATA__) and filter to this board.
    2) If none found, fall back to the board's RSS feed (board-only by design).
    Returns (pins, source) where source is 'json' | 'rss' | 'none'.
    """
    html = fetch_board_html(board_url)
    pins = extract_pins_from_html(html, board_url=board_url, max_pins=max_pins)
    if pins:
        return pins, "json"
    pins = fetch_board_rss(board_url, max_items=max_pins)
    if pins:
        return pins, "rss"
    return [], "none"


# ===============================
# Streamlit App — Clean UI
# ===============================

st.set_page_config(page_title="Pinterest Color & Trend Analyzer", layout="wide")
st.title("🎯 Pinterest Board — Color & Trend Analyzer")
st.caption("Paste a **public** Pinterest board URL and click Analyze. Board-only. Minimal UI.")

board_url = st.text_input(
    "Pinterest board URL",
    placeholder="https://www.pinterest.com/<username>/<board-slug>/",
)

with st.expander("Options", expanded=False):
    pin_limit = st.slider("Max pins to analyze", 20, 500, 120, step=10)
    palette_k = st.slider("Colors per image (KMeans)", 3, 8, 5)
    master_palette_k = st.slider("Master palette size (across board)", 5, 20, 10)
    grid_cols = st.slider("Pin gallery columns", 3, 10, 6)

analyze = st.button("Analyze")

if not analyze:
    st.stop()

if not board_url or "pinterest." not in urlparse(board_url).netloc:
    st.error("Please paste a valid public Pinterest board URL.")
    st.stop()

# ---------------------------
# Scrape (board-only, with fallback)
# ---------------------------
pins, source = scrape_board_boardonly(board_url, max_pins=pin_limit)
if not pins:
    st.error("No pins found. The board may be private, region-limited, or its data is unavailable.")
    st.stop()

pins_df = pd.DataFrame(pins)

# ---------------------------
# Palette extraction per pin
# ---------------------------
progress = st.progress(0)
records = []
for idx, row in pins_df.iterrows():
    arr = load_image_as_array(row["image_url"])
    if arr is None:
        continue

    # sample pixels for speed
    pixels = arr.reshape(-1, 3)
    if pixels.shape[0] > 6000:
        sel = np.random.RandomState(42).choice(pixels.shape[0], 6000, replace=False)
        pixels = pixels[sel]

    kmeans = KMeans(n_clusters=palette_k, n_init="auto", random_state=42)
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_.astype(float)
    hsv = rgb_to_hsv_np(centers)
    hexes = [hex_from_rgb(c) for c in centers]

    records.append(
        {
            "pin_id": row.get("pin_id"),
            "image_url": row["image_url"],
            "title": row.get("title", ""),
            "description": row.get("description", ""),
            "created_at": row.get("created_at"),
            "palette_hex": hexes,
            "palette_rgb": centers.tolist(),
            "palette_hsv": hsv.tolist(),
            "dominant_hex": hexes[0] if hexes else None,
        }
    )
    progress.progress(min(1.0, (len(records) / len(pins_df))))
progress.empty()

if not records:
    st.error("Images could not be processed. Try another public board.")
    st.stop()

colors_df = pd.DataFrame(records)

# Expand palettes to long form
pal_rows = []
for _, r in colors_df.iterrows():
    for i, hx in enumerate(r["palette_hex"]):
        rgb = r["palette_rgb"][i]
        hsv = r["palette_hsv"][i]
        pal_rows.append(
            {
                "pin_id": r["pin_id"],
                "image_url": r["image_url"],
                "title": r["title"],
                "hex": hx,
                "r": rgb[0],
                "g": rgb[1],
                "b": rgb[2],
                "h": hsv[0],
                "s": hsv[1],
                "v": hsv[2],
            }
        )
pal_df = pd.DataFrame(pal_rows)

# ---------------------------
# Master palette across the board
# ---------------------------
st.subheader("Master Color Palette")

all_rgb = pal_df[["r", "g", "b"]].to_numpy()
master = KMeans(n_clusters=master_palette_k, n_init="auto", random_state=42).fit(all_rgb)
centers = master.cluster_centers_.astype(float)
master_hex = [hex_from_rgb(c) for c in centers]
pal_df["cluster"] = master.predict(all_rgb)
cluster_counts = pal_df["cluster"].value_counts().sort_index()

# Swatches
cols = st.columns(min(6, master_palette_k))
for i, hx in enumerate(master_hex):
    with cols[i % len(cols)]:
        st.markdown(f"**{hx}**")
        st.markdown(
            f"""
            <div style='width:100%;height:42px;border-radius:8px;border:1px solid #ddd;background:{hx};'></div>
            <div style='font-size:12px;color:#666;'>share: {int(cluster_counts.get(i,0))}</div>
            """,
            unsafe_allow_html=True,
        )

# Bar chart of cluster shares (colored by actual hex)
share_df = pd.DataFrame(
    {
        "cluster": [f"C{i+1}" for i in range(master_palette_k)],
        "count": [int(cluster_counts.get(i, 0)) for i in range(master_palette_k)],
        "hex": master_hex,
    }
)
bar = (
    alt.Chart(share_df)
    .mark_bar(stroke="black", strokeWidth=0.25)
    .encode(
        x=alt.X("cluster:N", sort=None, title="Cluster"),
        y=alt.Y("count:Q", title="Frequency"),
        color=alt.Color("hex:N", scale=None, legend=None),
        tooltip=["cluster", "hex", "count"],
    )
    .properties(height=240)
)
st.altair_chart(bar, use_container_width=True)

# ---------------------------
# Pin Gallery (filterable)
# ---------------------------
st.subheader("Pin Gallery")

# Assign each pin to the nearest master cluster based on its dominant color
pin_clusters = []
dom_rgbs = colors_df["palette_rgb"].apply(lambda lst: lst[0] if (isinstance(lst, list) and len(lst) > 0) else [0, 0, 0])
for rgb in dom_rgbs:
    try:
        pin_clusters.append(int(master.predict([rgb])[0]))
    except Exception:
        pin_clusters.append(None)
colors_df["pin_cluster"] = pin_clusters

cluster_options = [f"C{i+1}" for i in range(master_palette_k)]
sel_clusters = st.multiselect("Filter by cluster", cluster_options, default=cluster_options)
sel_idx = [int(c[1:]) - 1 for c in sel_clusters]

gallery_df = colors_df[colors_df["pin_cluster"].isin(sel_idx)].copy()
if gallery_df.empty:
    st.info("No pins match the selected cluster filter.")
else:
    # grid positions
    gallery_df = gallery_df.reset_index(drop=True)
    gallery_df["idx"] = np.arange(len(gallery_df))
    gallery_df["row"] = (gallery_df["idx"] // grid_cols).astype(int)
    gallery_df["col"] = (gallery_df["idx"] % grid_cols).astype(int)

    # concise palette preview for tooltip
    gallery_df["palette_display"] = gallery_df["palette_hex"].apply(lambda xs: ", ".join(xs[:5]) if isinstance(xs, list) else "")

    # altair image grid (each cell is an image with a tooltip)
    # NOTE: Vega-Lite image mark expects a field named 'url'
    gallery_df["url"] = gallery_df["image_url"]

    thumb_w, thumb_h = 140, 140
    chart_height = (gallery_df["row"].max() + 1) * thumb_h if len(gallery_df) else 200
    chart_width = grid_cols * thumb_w

    img_chart = (
        alt.Chart(gallery_df)
        .mark_image(width=thumb_w, height=thumb_h)
        .encode(
            x=alt.X("col:O", axis=None),
            y=alt.Y("row:O", sort="descending", axis=None),
            url="url:N",
            tooltip=[
                alt.Tooltip("title:N", title="Title"),
                alt.Tooltip("dominant_hex:N", title="Dominant"),
                alt.Tooltip("palette_display:N", title="Palette (top 5)"),
            ],
        )
        .properties(width=chart_width, height=chart_height)
    )
    st.altair_chart(img_chart, use_container_width=True)

# ---------------------------
# HSV insights — color-true
# ---------------------------
st.subheader("Hue / Saturation / Value Insights")

# Hue histogram (bin & color bars by bin hue)
pal_df["h_deg"] = (pal_df["h"] * 360.0).round(1)
bins = np.arange(0, 361, 10)
labels = (bins[:-1] + bins[1:]) / 2
pal_df["h_bin"] = pd.cut(pal_df["h_deg"], bins=bins, include_lowest=True, labels=labels)
h_bin_df = pal_df.groupby("h_bin").size().reset_index(name="count")
h_bin_df["h_mid"] = h_bin_df["h_bin"].astype(float)
h_bin_df["h_color"] = h_bin_df["h_mid"].apply(lambda d: f"hsl({int(d)}, 90%, 50%)")

hue_hist = (
    alt.Chart(h_bin_df)
    .mark_bar(stroke="black", strokeWidth=0.25)
    .encode(
        x=alt.X("h_mid:Q", title="Hue (°)"),
        y=alt.Y("count:Q", title="Count"),
        color=alt.Color("h_color:N", scale=None, legend=None),
        tooltip=["h_mid", "count"],
    )
    .properties(height=240)
)
st.altair_chart(hue_hist, use_container_width=True)

# Saturation vs Value scatter (dot = actual hex)
sv_scatter = (
    alt.Chart(pal_df)
    .mark_circle(size=60, stroke="black", strokeWidth=0.15)
    .encode(
        x=alt.X("s:Q", title="Saturation"),
        y=alt.Y("v:Q", title="Value (Brightness)"),
        color=alt.Color("hex:N", scale=None, legend=None),
        tooltip=["hex", "title"],
    )
    .properties(height=300)
)
st.altair_chart(sv_scatter, use_container_width=True)

# ---------------------------
# Color Share (Waffle)
# ---------------------------
st.subheader("Color Share (Waffle)")
total = int(share_df["count"].sum())
if total > 0:
    N = 100
    raw = (share_df["count"] / total * N).round().astype(int)
    # balance rounding
    diff = N - int(raw.sum())
    if diff != 0:
        adjust_idx = int(np.argmax(share_df["count"])) if diff > 0 else int(np.argmin(share_df["count"]))
        raw.iloc[adjust_idx] = raw.iloc[adjust_idx] + diff
    tiles = []
    cols_n = 10
    rows_n = int(math.ceil(N / cols_n))
    k = 0
    for cluster, n_tiles, hexv in zip(share_df["cluster"], raw, share_df["hex"]):
        for _ in range(int(n_tiles)):
            r = k // cols_n
            c = k % cols_n
            tiles.append({"row": rows_n - 1 - r, "col": c, "cluster": cluster, "hex": hexv})
            k += 1
    waffle_df = pd.DataFrame(tiles)
    waffle = (
        alt.Chart(waffle_df)
        .mark_rect(stroke="white", strokeWidth=0.5)
        .encode(
            x=alt.X("col:O", axis=None),
            y=alt.Y("row:O", sort="descending", axis=None),
            color=alt.Color("hex:N", scale=None, legend=None),
            tooltip=["cluster", "hex"],
        )
        .properties(height=220)
    )
    st.altair_chart(waffle, use_container_width=True)
else:
    st.info("Not enough color data to render waffle.")

# ---------------------------
# Per-Cluster Distributions
# ---------------------------
cluster_hex_map = {i: master_hex[i] for i in range(len(master_hex))}
pal_df["cluster_hex"] = pal_df["cluster"].map(cluster_hex_map)

st.subheader("Per-Cluster Brightness (V) Distribution")
v_box = (
    alt.Chart(pal_df)
    .mark_boxplot()
    .encode(
        x=alt.X("cluster:N", title="Cluster"),
        y=alt.Y("v:Q", title="Brightness (V)"),
        color=alt.Color("cluster_hex:N", scale=None, legend=None),
    )
    .properties(height=280)
)
st.altair_chart(v_box, use_container_width=True)

st.subheader("Per-Cluster Saturation (S) Distribution")
s_box = (
    alt.Chart(pal_df)
    .mark_boxplot()
    .encode(
        x=alt.X("cluster:N", title="Cluster"),
        y=alt.Y("s:Q", title="Saturation (S)"),
        color=alt.Color("cluster_hex:N", scale=None, legend=None),
    )
    .properties(height=280)
)
st.altair_chart(s_box, use_container_width=True)

# ---------------------------
# Hue × Value Heatmap (Average Color)
# ---------------------------
st.subheader("Hue × Value Heatmap (Average Color)")
h_bins = np.arange(0, 361, 15)
v_bins = np.linspace(0, 1, 11)
pal_df["h_bin2"] = pd.cut((pal_df["h"] * 360.0), bins=h_bins, include_lowest=True, labels=((h_bins[:-1] + h_bins[1:]) / 2))
pal_df["v_bin2"] = pd.cut(pal_df["v"], bins=v_bins, include_lowest=True, labels=((v_bins[:-1] + v_bins[1:]) / 2))
hv = (
    pal_df.groupby(["h_bin2", "v_bin2"])
    .agg(count=("hex", "size"), r_mean=("r", "mean"), g_mean=("g", "mean"), b_mean=("b", "mean"))
    .reset_index()
    .dropna()
)
hv["hex"] = hv.apply(lambda r: hex_from_rgb((r["r_mean"], r["g_mean"], r["b_mean"])), axis=1)
hv["h_mid"] = hv["h_bin2"].astype(float)
hv["v_mid"] = hv["v_bin2"].astype(float)

heat = (
    alt.Chart(hv)
    .mark_rect(stroke="black", strokeWidth=0.1)
    .encode(
        x=alt.X("h_mid:Q", title="Hue (°)"),
        y=alt.Y("v_mid:Q", title="Value (Brightness)"),
        color=alt.Color("hex:N", scale=None, legend=None),
        tooltip=["h_mid", "v_mid", "count"],
    )
    .properties(height=320)
)
st.altair_chart(heat, use_container_width=True)

# ---------------------------
# Dominant Color Sequence (by pin order)
# ---------------------------
st.subheader("Dominant Color Sequence")
colors_df["order"] = np.arange(len(colors_df))

def _dom_h_deg(pal):
    try:
        return float(pal[0][0]) * 360.0
    except Exception:
        return np.nan

colors_df["dom_h_deg"] = colors_df["palette_hsv"].apply(_dom_h_deg)

seq_line = alt.Chart(colors_df).mark_line(color="#888").encode(
    x=alt.X("order:Q", title="Pin order (scrape)"),
    y=alt.Y("dom_h_deg:Q", title="Dominant Hue (°)"),
)
seq_pts = alt.Chart(colors_df).mark_point(size=60).encode(
    x="order:Q",
    y="dom_h_deg:Q",
    color=alt.Color("dominant_hex:N", scale=None, legend=None),
    tooltip=["title", "dominant_hex", "order", "dom_h_deg"],
)
st.altair_chart(seq_line + seq_pts, use_container_width=True)

# ---------------------------
# Diagnostics
# ---------------------------
with st.expander("🔧 Diagnostics"):
    st.write(
        {
            "board_url": board_url,
            "pins_found": int(len(pins_df)),
            "method": source,  # 'json' or 'rss' or 'none'
        }
    )
