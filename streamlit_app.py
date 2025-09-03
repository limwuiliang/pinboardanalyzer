# streamlit_app.py
# -------------------------------------------------------------
# Pinterest Board Color & Trend Analyzer — URL-only, clean UI
# -------------------------------------------------------------
# What this version does
# - User pastes a **public Pinterest board URL** → app scrapes the page
# - Extracts pin image URLs (best-effort) + any titles/descriptions found
# - Runs color analysis (KMeans palettes), HSV insights, keyword bars
# - Minimal UI; options tucked into an expander
# - CSV export of analyzed colors + PNG of master palette
# -------------------------------------------------------------

import io
import re
import math
import json
from urllib.parse import urlparse

import requests
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

import streamlit as st
import altair as alt

# ===============================
# Utilities
# ===============================

def hex_from_rgb(rgb_tuple):
    return '#%02x%02x%02x' % tuple(int(max(0, min(255, c))) for c in rgb_tuple)

def rgb_to_hsv_np(rgb):
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
        r = requests.get(url, timeout=timeout, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
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
            img = img.convert('RGB')
            w, h = img.size
            scale = max(w, h) / max_side if max(w, h) > max_side else 1
            if scale > 1:
                img = img.resize((int(w/scale), int(h/scale)), Image.LANCZOS)
            return np.array(img)
    except Exception:
        return None

# ===============================
# Scraper (URL-only, best-effort)
# ===============================

PINTEREST_IMG_RE = re.compile(r'https?://i\.pinimg\.com/[^"\\\s>]+\.(?:jpg|jpeg|png|webp)')

@st.cache_data(show_spinner=True)
def fetch_board_html(board_url: str) -> str:
    r = requests.get(board_url, timeout=20, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9'
    })
    r.raise_for_status()
    return r.text

@st.cache_data(show_spinner=False)
def extract_pins_from_html(html: str, max_pins: int = 200):
    pins = []

    # 1) Try the embedded JSON state first (script#__PWS_DATA__)
    m = re.search(r'<script[^>]+id="__PWS_DATA__"[^>]*>(.*?)</script>', html, flags=re.DOTALL|re.IGNORECASE)
    if m:
        try:
            blob = m.group(1)
            data = json.loads(blob)

            def walk(o):
                if isinstance(o, dict):
                    # Find images object shapes like { images: { orig: {url}, 736x: {url}, ... }, title, description, created_at }
                    images = o.get('images') or (o.get('image') if isinstance(o.get('image'), dict) else None)
                    img_url = None
                    if isinstance(images, dict):
                        for key in ('orig','736x','474x','170x','small','medium','large'):
                            if key in images and isinstance(images[key], dict) and images[key].get('url'):
                                img_url = images[key]['url']
                                break
                    title = o.get('title') or o.get('grid_title') or o.get('alt_text') or ''
                    description = o.get('description') or o.get('grid_description') or ''
                    created_at = o.get('created_at') or o.get('created') or None
                    pin_id = o.get('id') or o.get('pin_id')
                    if img_url:
                        pins.append({
                            'pin_id': str(pin_id) if pin_id else None,
                            'title': title,
                            'description': description,
                            'created_at': created_at,
                            'image_url': img_url,
                        })
                    for v in o.values():
                        walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)
            walk(data)
        except Exception:
            pass

    # 2) Fallback: regex any pinimg CDN URLs from the whole HTML
    if not pins:
        imgs = list({u for u in PINTEREST_IMG_RE.findall(html)})
        pins = [{'pin_id': None, 'title': '', 'description': '', 'created_at': None, 'image_url': u} for u in imgs]

    # Deduplicate by image_url and trim
    dedup, seen = [], set()
    for p in pins:
        url = p.get('image_url')
        if url and url not in seen:
            dedup.append(p)
            seen.add(url)
        if len(dedup) >= max_pins:
            break
    return dedup

@st.cache_data(show_spinner=True)
def scrape_board(board_url: str, max_pins: int = 200):
    html = fetch_board_html(board_url)
    return extract_pins_from_html(html, max_pins=max_pins)

# ===============================
# Text tokens
# ===============================

STOPWORDS = set((
    'the','a','an','and','or','of','for','to','in','on','at','with','by','from','this','that','is','are','it','its','your','you','i','we','they','our'
))

def tokenize_texts(texts):
    docs = [re.sub(r"[^\w\s]", " ", (t or "")).lower() for t in texts]
    docs = [re.sub(r"\s+", " ", d).strip() for d in docs]
    return docs

def top_tokens_series(texts, top_k=25, min_df=1):
    docs = tokenize_texts(texts)
    if not any(docs):
        return pd.Series(dtype=int)
    vectorizer = CountVectorizer(stop_words=list(STOPWORDS), min_df=min_df)
    X = vectorizer.fit_transform(docs)
    sums = np.asarray(X.sum(axis=0)).ravel()
    tokens = np.array(vectorizer.get_feature_names_out())
    df = pd.DataFrame({'token': tokens, 'count': sums})
    df = df.sort_values('count', ascending=False).head(top_k)
    return df.set_index('token')['count']

# ===============================
# Streamlit App — Clean UI
# ===============================

st.set_page_config(page_title='Pinterest Color & Trend Analyzer', layout='wide')
st.title('🎯 Pinterest Board — Color & Trend Analyzer')
st.caption('Paste a **public** Pinterest board URL and click Analyze.')

board_url = st.text_input('Pinterest board URL', placeholder='https://www.pinterest.com/<username>/<board-slug>/')

with st.expander('Options', expanded=False):
    pin_limit = st.slider('Max pins to analyze', 20, 500, 120, step=10)
    palette_k = st.slider('Colors per image (KMeans)', 3, 8, 5)
    master_palette_k = st.slider('Master palette size (across board)', 5, 20, 10)

analyze = st.button('Analyze')

if not analyze:
    st.stop()

if not board_url or 'pinterest.' not in urlparse(board_url).netloc:
    st.error('Please paste a valid public Pinterest board URL.')
    st.stop()

# ---------------------------
# Scrape
# ---------------------------

try:
    pins = scrape_board(board_url, max_pins=pin_limit)
except Exception as e:
    st.exception(e)
    st.stop()

if not pins:
    st.error('No pins found. The board may be private or blocked. Make sure it is public and try again.')
    st.stop()

pins_df = pd.DataFrame(pins)

# ---------------------------
# Palette extraction per pin
# ---------------------------

progress = st.progress(0)
records = []
for idx, row in pins_df.iterrows():
    arr = load_image_as_array(row['image_url'])
    if arr is None:
        continue
    kmeans = KMeans(n_clusters=palette_k, n_init='auto', random_state=42)
    pixels = arr.reshape(-1, 3)
    if pixels.shape[0] > 6000:
        sel = np.random.RandomState(42).choice(pixels.shape[0], 6000, replace=False)
        pixels = pixels[sel]
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_.astype(int)
    hsv = rgb_to_hsv_np(centers)
    hexes = [hex_from_rgb(c) for c in centers]
    records.append({
        'pin_id': row.get('pin_id'),
        'image_url': row['image_url'],
        'title': row.get('title',''),
        'description': row.get('description',''),
        'created_at': row.get('created_at'),
        'palette_hex': hexes,
        'palette_rgb': centers.tolist(),
        'palette_hsv': hsv.tolist(),
        'dominant_hex': hexes[0] if hexes else None,
    })
    progress.progress(min(1.0, (len(records) / len(pins_df))))
progress.empty()

if not records:
    st.error('Images could not be processed. Try another public board.')
    st.stop()

colors_df = pd.DataFrame(records)

# Expand palettes to long form
pal_rows = []
for _, r in colors_df.iterrows():
    for i, hx in enumerate(r['palette_hex']):
        rgb = r['palette_rgb'][i]
        hsv = r['palette_hsv'][i]
        pal_rows.append({
            'pin_id': r['pin_id'],
            'image_url': r['image_url'],
            'title': r['title'],
            'hex': hx,
            'r': rgb[0], 'g': rgb[1], 'b': rgb[2],
            'h': hsv[0], 's': hsv[1], 'v': hsv[2],
        })
pal_df = pd.DataFrame(pal_rows)

# ---------------------------
# Master palette across the board
# ---------------------------

st.subheader('Master Color Palette')
all_rgb = pal_df[['r','g','b']].to_numpy()
master = KMeans(n_clusters=master_palette_k, n_init='auto', random_state=42).fit(all_rgb)
centers = master.cluster_centers_.astype(int)
master_hex = [hex_from_rgb(c) for c in centers]
pal_df['cluster'] = master.predict(all_rgb)
cluster_counts = pal_df['cluster'].value_counts().sort_index()

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

share_df = pd.DataFrame({
    'cluster': [f'C{i+1}' for i in range(master_palette_k)],
    'count': [int(cluster_counts.get(i,0)) for i in range(master_palette_k)],
    'hex': master_hex,
})
bar = alt.Chart(share_df).mark_bar(stroke='black', strokeWidth=0.25).encode(
    x=alt.X('cluster:N', sort=None, title='Cluster'),
    y=alt.Y('count:Q', title='Frequency'),
    color=alt.Color('hex:N', scale=None, legend=None),
    tooltip=['cluster','hex','count']
).properties(height=240)
st.altair_chart(bar, use_container_width=True)

# ---------------------------
# HSV insights
# ---------------------------

st.subheader('Hue / Saturation / Value Insights')
pal_df['h_deg'] = (pal_df['h'] * 360.0).round(1)

# Pre-bin hue so we can paint bars with actual hue colors
pal_df['h_deg'] = (pal_df['h'] * 360.0).round(1)
bins = np.arange(0, 361, 10)
labels = (bins[:-1] + bins[1:]) / 2
pal_df['h_bin'] = pd.cut(pal_df['h_deg'], bins=bins, include_lowest=True, labels=labels)
h_bin_df = pal_df.groupby('h_bin').size().reset_index(name='count')
h_bin_df['h_mid'] = h_bin_df['h_bin'].astype(float)
h_bin_df['h_color'] = h_bin_df['h_mid'].apply(lambda d: f'hsl({int(d)}, 90%, 50%)')

hue_hist = alt.Chart(h_bin_df).mark_bar(stroke='black', strokeWidth=0.25).encode(
    x=alt.X('h_mid:Q', title='Hue (°)'),
    y=alt.Y('count:Q', title='Count'),
    color=alt.Color('h_color:N', scale=None, legend=None),
    tooltip=['h_mid','count']
).properties(height=240)
st.altair_chart(hue_hist, use_container_width=True)

sv_scatter = alt.Chart(pal_df).mark_circle(size=60, stroke='black', strokeWidth=0.15).encode(
    x=alt.X('s:Q', title='Saturation'),
    y=alt.Y('v:Q', title='Value (Brightness)'),
    color=alt.Color('hex:N', scale=None, legend=None),
    tooltip=['hex','title']
).properties(height=300)
st.altair_chart(sv_scatter, use_container_width=True)

# ---------------------------
# Keyword signals (from titles/descriptions if available)
# ---------------------------

st.subheader('Keyword Signals')
texts = (colors_df['title'].fillna('') + ' ' + colors_df['description'].fillna('')).tolist()
kw_series = top_tokens_series(texts, top_k=25, min_df=1)
if not kw_series.empty:
    kw_df = kw_series.reset_index()
    kw_df.columns = ['token','count']
    kw_chart = alt.Chart(kw_df).mark_bar().encode(
        x=alt.X('count:Q', title='Count'),
        y=alt.Y('token:N', sort='-x', title='Token'),
        tooltip=['token','count']
    ).properties(height=380)
    st.altair_chart(kw_chart, use_container_width=True)
else:
    st.info('No meaningful keywords found (board HTML may not include titles/descriptions).')

# ---------------------------
# Data & exports
# ---------------------------

st.subheader('Data & Exports')
flat_rows = []
for _, r in colors_df.iterrows():
    row = {
        'pin_id': r['pin_id'], 'title': r['title'], 'description': r['description'], 'created_at': r['created_at'], 'image_url': r['image_url']
    }
    for i, hx in enumerate(r['palette_hex'][:8]):
        row[f'color_{i+1}_hex'] = hx
    flat_rows.append(row)
flat_df = pd.DataFrame(flat_rows)

st.dataframe(flat_df, use_container_width=True)

csv = flat_df.to_csv(index=False).encode('utf-8')
st.download_button('Download analyzed data (CSV)', data=csv, file_name='pinterest_color_trends.csv', mime='text/csv')

# Master palette PNG
sw, sh, cols_n = 220, 60, 2
rows = math.ceil(len(master_hex) / cols_n)
img = Image.new('RGB', (sw*cols_n, sh*rows), (255,255,255))
draw_idx = 0
for r in range(rows):
    for c in range(cols_n):
        if draw_idx >= len(master_hex):
            break
        hx = master_hex[draw_idx]
        rgb = tuple(int(hx[i:i+2], 16) for i in (1,3,5))
        block = Image.new('RGB', (sw, sh), rgb)
        img.paste(block, (c*sw, r*sh))
        draw_idx += 1
buf = io.BytesIO()
img.save(buf, format='PNG')
palette_png = buf.getvalue()
st.download_button('Download master palette (PNG)', data=palette_png, file_name='board_master_palette.png', mime='image/png')

# ---------------------------
# Notes
# ---------------------------
with st.expander('ℹ️ Notes'):
    st.markdown(
        """
        - This app analyzes **public** boards. Private or region-restricted boards won't load.
        - Pinterest's HTML/JSON structure can change; the scraper is **best-effort**. If a board doesn't parse, try another board.
        - Colors use **KMeans** clustering over sampled pixels for speed/quality balance.
        - For more consistent metadata (titles, dates), consider adding an authenticated API path in a future version.
        """
    )
