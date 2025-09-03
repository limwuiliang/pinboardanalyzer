# streamlit_app.py
# -------------------------------------------------------------
# Pinterest Board Color Analyzer — board-only, resilient fetch
# -------------------------------------------------------------

import io, re, json, html, math, base64, xml.etree.ElementTree as ET
from urllib.parse import urlparse, urljoin

import requests, numpy as np, pandas as pd
from PIL import Image
from sklearn.cluster import KMeans

import streamlit as st
import altair as alt

# ===============================
# Helpers
# ===============================

def hex_from_rgb(rgb_tuple):
    r, g, b = [int(max(0, min(255, round(c)))) for c in rgb_tuple]
    return f"#{r:02x}{g:02x}{b:02x}"

def rgb_to_hsv_np(rgb):
    rgb = np.asarray(rgb, dtype=np.float32) / 255.0
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    mx, mn = np.max(rgb, axis=-1), np.min(rgb, axis=-1)
    diff = mx - mn
    h = np.zeros_like(mx)
    mask = diff != 0
    r_eq, g_eq, b_eq = (mx==r)&mask, (mx==g)&mask, (mx==b)&mask
    h[r_eq] = ((g[r_eq]-b[r_eq])/diff[r_eq]) % 6
    h[g_eq] = ((b[g_eq]-r[g_eq])/diff[g_eq]) + 2
    h[b_eq] = ((r[b_eq]-g[b_eq])/diff[b_eq]) + 4
    h = (h/6.0) % 1.0
    s = np.where(mx==0, 0, diff/mx)
    v = mx
    return np.stack([h,s,v], axis=-1)

def _get_cookie(key: str, cookie: str | None):
    if not cookie: return None
    m = re.search(rf"(?:^|;\s*){re.escape(key)}=([^;]+)", cookie)
    return m.group(1) if m else None

def build_headers(cookie: str | None, accept="*/*", referer="https://www.pinterest.com/"):
    h = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept": accept,
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": referer,
        "X-Requested-With": "XMLHttpRequest",
    }
    if cookie:
        h["Cookie"] = cookie
        csrf = _get_cookie("csrftoken", cookie)
        if csrf: h["X-CSRFToken"] = csrf
    return h

PIN_SIZE_PATTERN = re.compile(r"/(orig(?:inals)?|[0-9]{3,4}x)/", re.IGNORECASE)
def rewrite_pinimg_size(u: str, size: str) -> str:
    if "i.pinimg.com" not in u: return u
    if PIN_SIZE_PATTERN.search(u): return PIN_SIZE_PATTERN.sub(f"/{size}/", u)
    parts = u.split("/")
    if len(parts) >= 5:
        parts.insert(4, size)
        return "/".join(parts)
    return u

def canonicalize_pin_url(u: str) -> str:
    if not isinstance(u, str): return ""
    core = u.split("?")[0]
    if "i.pinimg.com" in core:
        core = rewrite_pinimg_size(core, "736x")
    return core

def _normalize_board_path(u: str):
    try:
        p = urlparse(u)
        parts = [x for x in (p.path or "").strip("/").split("/") if x]
        return f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else None
    except Exception:
        return None

# Convert numpy image -> small data URI (for Altair mark_image; avoids CORS)
def array_to_data_uri(arr: np.ndarray, max_side: int = 112, fmt: str = "JPEG", quality: int = 78) -> str:
    try:
        img = Image.fromarray(arr.astype(np.uint8))
        w, h = img.size
        scale = max(w, h) / max_side if max(w, h) > max_side else 1
        if scale > 1:
            img = img.resize((int(w/scale), int(h/scale)), Image.LANCZOS)
        bio = io.BytesIO()
        if fmt.upper() == "PNG":
            img.save(bio, "PNG", optimize=True); mime = "image/png"
        else:
            img.save(bio, "JPEG", quality=quality, optimize=True, subsampling=2); mime = "image/jpeg"
        b64 = base64.b64encode(bio.getvalue()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return ""

# ===============================
# Image IO
# ===============================

@st.cache_data(show_spinner=False)
def fetch_image_bytes(url, cookie=None, timeout=15):
    try:
        r = requests.get(url, timeout=timeout,
                         headers=build_headers(cookie, accept="image/avif,image/webp,image/*,*/*;q=0.8"))
        r.raise_for_status()
        return r.content
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_image_as_array(url, cookie=None, max_side=512):
    candidates = [url]
    if "i.pinimg.com" in url:
        for size in ("736x","474x","236x"):
            v = rewrite_pinimg_size(url, size)
            if v not in candidates: candidates.append(v)
    for u in candidates:
        data = fetch_image_bytes(u, cookie=cookie)
        if not data: continue
        try:
            with Image.open(io.BytesIO(data)) as img:
                img = img.convert("RGB")
                w, h = img.size
                scale = max(w,h)/max_side if max(w,h) > max_side else 1
                if scale > 1:
                    img = img.resize((int(w/scale), int(h/scale)), Image.LANCZOS)
                return np.array(img)
        except Exception:
            continue
    return None

# ===============================
# Scrapers (board-only)
# ===============================

PINTEREST_IMG_RE = re.compile(r'https?://i\.pinimg\.com/[^\s">]+\.(?:jpg|jpeg|png|webp)')

@st.cache_data(show_spinner=True)
def fetch_board_html(url: str, cookie=None) -> str:
    r = requests.get(url, timeout=20, headers=build_headers(cookie, accept="text/html,application/xhtml+xml"))
    r.raise_for_status()
    return r.text

def extract_board_id_pin_total(html_text: str):
    m = re.search(r'<script[^>]+id="__PWS_DATA__"[^>]*>(.*?)</script>', html_text, flags=re.DOTALL|re.IGNORECASE)
    if not m: return None, None
    blob = m.group(1)
    try:
        data = json.loads(blob)
        txt = json.dumps(data)
        bid = re.search(r'"board_id"\s*:\s*"(\d+)"', txt)
        total = re.search(r'"pin_count"\s*:\s*(\d+)', txt)
        return (bid.group(1) if bid else None), (int(total.group(1)) if total else None)
    except Exception:
        return None, None

@st.cache_data(show_spinner=False)
def extract_pins_from_html(html_text: str, board_url: str, max_pins: int = 200):
    target_path = _normalize_board_path(board_url)
    m = re.search(r'<script[^>]+id="__PWS_DATA__"[^>]*>(.*?)</script>', html_text, flags=re.DOTALL|re.IGNORECASE)
    if not m: return [], None
    blob = m.group(1); board_id = None
    try:
        data = json.loads(blob)
        t = json.dumps(data)
        mm = re.search(r'"board_id"\s*:\s*"(\d+)"', t)
        if mm: board_id = mm.group(1)
    except Exception:
        return [], board_id

    pins = []
    def walk(o):
        if isinstance(o, dict):
            images = o.get("images") or (o.get("image") if isinstance(o.get("image"), dict) else None)
            img_url = None
            if isinstance(images, dict):
                for key in ("orig","736x","474x","236x","170x","small","medium","large"):
                    d = images.get(key)
                    if isinstance(d, dict) and d.get("url"):
                        img_url = d["url"]; break
            belongs = False
            b = o.get("board") or {}
            burl = b.get("url") or o.get("board_url") or o.get("url")
            if isinstance(burl, str) and target_path:
                belongs = (_normalize_board_path(burl) or "") == target_path
            if img_url and belongs:
                pins.append({
                    "pin_id": str(o.get("id") or o.get("pin_id") or "") or None,
                    "title": o.get("title") or o.get("grid_title") or o.get("alt_text") or "",
                    "description": o.get("description") or o.get("grid_description") or "",
                    "created_at": o.get("created_at") or o.get("created") or None,
                    "image_url": img_url,
                })
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    try: walk(data)
    except Exception: pass

    dedup, seen = [], set()
    for p in pins:
        u = p.get("image_url")
        if u and u not in seen:
            dedup.append(p); seen.add(u)
        if len(dedup) >= max_pins: break
    return dedup, board_id

# Public widgets
@st.cache_data(show_spinner=True)
def fetch_board_pidgets(board_url: str, want: int = 400, page_size: int = 100, max_pages: int = 20):
    path = _normalize_board_path(board_url)
    if not path: return [], 0
    base = f"https://widgets.pinterest.com/v3/pidgets/boards/{path}/pins/"
    params = {"page_size": str(page_size)}
    headers = build_headers(cookie=None, accept="application/json", referer=f"https://www.pinterest.com/{path}/")

    pins, seen_ids, seen_urls, pages, bookmark = [], set(), set(), 0, None
    while len(pins) < want and pages < max_pages:
        q = params.copy()
        if bookmark: q["bookmark"] = bookmark
        try:
            r = requests.get(base, params=q, headers=headers, timeout=20)
            if r.status_code != 200: break
            obj = r.json(); pages += 1
            data = (obj.get("data") or {})
            bookmark = data.get("bookmark")
            for o in (data.get("pins") or []):
                images = o.get("images") or {}
                img_url = None
                for sz in ("736x","564x","474x","236x","orig","170x"):
                    d = images.get(sz)
                    if isinstance(d, dict) and d.get("url"):
                        img_url = d["url"]; break
                if not img_url: continue
                pid = str(o.get("id")) if o.get("id") else None
                canon = canonicalize_pin_url(img_url)
                if pid:
                    if pid in seen_ids: continue
                    seen_ids.add(pid)
                else:
                    if canon in seen_urls: continue
                    seen_urls.add(canon)
                pins.append({"pin_id": pid, "title": o.get("title") or "", "description": "",
                             "created_at": o.get("created_at") or None, "image_url": img_url})
            if not bookmark: break
        except Exception:
            break
    return pins, pages

# RSS (older pins; parse <guid> / <link> for pin_id)
def _parse_rss_batch(xml_text: str):
    try: root = ET.fromstring(xml_text)
    except Exception: return [], None
    ns = {"media":"http://search.yahoo.com/mrss/","atom":"http://www.w3.org/2005/Atom"}
    items = root.findall(".//item")
    pins = []
    for it in items:
        title = (it.findtext("title") or "")
        link_txt = (it.findtext("link") or "")
        guid_txt = (it.findtext("guid") or "")
        pid = None
        for txt in (link_txt, guid_txt):
            m = re.search(r"/pin/(\d+)", txt) or re.search(r"(\d{7,})", txt)
            if m: pid = m.group(1); break
        created_at = it.findtext("pubDate") or None
        img_url = None
        mc = it.find("media:content", ns); mt = it.find("media:thumbnail", ns); enc = it.find("enclosure")
        if mc is not None and mc.attrib.get("url"): img_url = mc.attrib["url"]
        elif mt is not None and mt.attrib.get("url"): img_url = mt.attrib["url"]
        elif enc is not None and enc.attrib.get("url"): img_url = enc.attrib["url"]
        if not img_url:
            desc = it.findtext("description") or ""
            mm = PINTEREST_IMG_RE.search(desc)
            if mm: img_url = mm.group(0)
        if img_url:
            pins.append({"pin_id": pid, "title": title, "description": "",
                         "created_at": created_at, "image_url": img_url})
    next_href = None
    for link in root.findall(".//atom:link", ns):
        if link.attrib.get("rel") == "next" and link.attrib.get("href"):
            next_href = link.attrib["href"]; break
    return pins, next_href

@st.cache_data(show_spinner=True)
def fetch_board_rss_paginated(board_url: str, max_items: int = 400, max_pages: int = 12):
    try:
        p = urlparse(board_url)
        parts = [x for x in (p.path or "").strip("/").split("/") if x]
        if len(parts) < 2: return [], 0
        base = f"{p.scheme}://{p.netloc}/{parts[0]}/{parts[1]}.rss"
        all_pins, seen, pages = [], set(), 0
        for start in (f"{base}?num=100", base):
            next_url = start
            while next_url and pages < max_pages and len(all_pins) < max_items:
                r = requests.get(next_url, timeout=20, headers=build_headers(None, accept="application/rss+xml, application/xml;q=0.9, */*;q=0.8"))
                if r.status_code != 200: break
                batch, next_href = _parse_rss_batch(r.text); pages += 1
                for pin in batch:
                    u = canonicalize_pin_url(pin.get("image_url"))
                    if u and u not in seen:
                        pin["image_url"] = u
                        all_pins.append(pin); seen.add(u)
                        if len(all_pins) >= max_items: break
                next_url = urljoin(next_url, next_href) if next_href else None
            if len(all_pins) >= max_items: break
        return all_pins, pages
    except Exception:
        return [], 0

# Cookie-enabled feed (lets us go past ~50 on many boards)
@st.cache_data(show_spinner=True)
def fetch_board_feed(board_url: str, board_id: str, cookie: str, want: int = 400, page_size: int = 100, max_pages: int = 12):
    path = _normalize_board_path(board_url) or ""
    source_url = f"/{path}/"
    endpoint = "https://www.pinterest.com/resource/BoardFeedResource/get/"
    pins, seen_ids, bookmark, pages = [], set(), None, 0

    def params(book):
        options = {"board_id": board_id, "page_size": page_size}
        if book: options["bookmarks"] = [book]
        data = {"options": options, "context": {}}
        return {"source_url": source_url, "data": json.dumps(data, separators=(",",":"))}

    while len(pins) < want and pages < max_pages:
        try:
            r = requests.get(endpoint, params=params(bookmark), headers=build_headers(cookie, accept="application/json"), timeout=20)
            if r.status_code != 200: break
            obj = r.json(); pages += 1
            rr = obj.get("resource_response") or {}
            data = rr.get("data") or []; bookmark = rr.get("bookmark")
            for o in data:
                images = o.get("images") or (o.get("image") if isinstance(o.get("image"), dict) else None)
                img_url = None
                if isinstance(images, dict):
                    for key in ("orig","736x","474x","236x","170x","small","medium","large"):
                        d = images.get(key)
                        if isinstance(d, dict) and d.get("url"): img_url = d["url"]; break
                if not img_url: continue
                if str(o.get("board_id") or (o.get("board") or {}).get("id")) != str(board_id): continue
                pid = str(o.get("id")) if o.get("id") else None
                if pid and pid in seen_ids: continue
                if pid: seen_ids.add(pid)
                pins.append({"pin_id": pid, "title": o.get("title") or "", "description": "",
                             "created_at": o.get("created_at") or None, "image_url": img_url})
            if not bookmark: break
        except Exception:
            break
    return pins, pages

@st.cache_data(show_spinner=True)
def fetch_board_html_pages(board_url: str, max_items: int = 400, max_pages: int = 10):
    pins, seen, pages = [], set(), 0
    urls = [board_url.rstrip("/")] + [f"{board_url.rstrip('/')}/?page={i}" for i in range(2, max_pages+1)]
    for u in urls:
        try:
            html_text = fetch_board_html(u, cookie=None)
            page_pins, _ = extract_pins_from_html(html_text, board_url, max_pins=max_items*2)
            pages += 1
            for p in page_pins:
                img = canonicalize_pin_url(p.get("image_url"))
                if img and img not in seen:
                    p["image_url"] = img
                    pins.append(p); seen.add(img)
                    if len(pins) >= max_items: return pins, pages
        except Exception:
            continue
    return pins, pages

# Union orchestrator (board-only, keeps distinct pin_ids)
@st.cache_data(show_spinner=True)
def scrape_board_boardonly(board_url: str, cookie=None, max_pins: int = 400):
    html_doc = fetch_board_html(board_url, cookie=cookie)
    board_id, board_pin_total = extract_board_id_pin_total(html_doc)
    from_html, _ = extract_pins_from_html(html_doc, board_url, max_pins=max_pins*2)

    acc = []; seen_ids, url_to_ids, anon_urls = set(), {}, set()
    def add_many(rows):
        for p in rows:
            if len(acc) >= max_pins: break
            pid = str(p.get("pin_id")) if p.get("pin_id") else None
            url = canonicalize_pin_url(p.get("image_url"))
            if not url: continue
            if pid:
                if pid in seen_ids: continue
                seen_ids.add(pid); url_to_ids.setdefault(url, set()).add(pid)
            else:
                if url in url_to_ids and url_to_ids[url]: continue
                if url in anon_urls: continue
                anon_urls.add(url); url_to_ids.setdefault(url, set())
            p["image_url"] = url
            acc.append(p)

    sources = []
    if from_html: add_many(from_html); sources.append("json")
    if cookie and board_id:  # only when cookie present
        feed_pins, fp = fetch_board_feed(board_url, board_id, cookie, want=max_pins*2, page_size=100, max_pages=12)
        if feed_pins: add_many(feed_pins); sources.append(f"feed({fp})")
    pidgets_pins, pp = fetch_board_pidgets(board_url, want=max_pins*2, page_size=100, max_pages=20)
    if pidgets_pins: add_many(pidgets_pins); sources.append(f"pidgets({pp})")
    rss_pins, rp = fetch_board_rss_paginated(board_url, max_items=max_pins*2, max_pages=12)
    if rss_pins: add_many(rss_pins); sources.append(f"rss({rp})")
    html2_pins, hp = fetch_board_html_pages(board_url, max_items=max_pins*2, max_pages=10)
    if html2_pins: add_many(html2_pins); sources.append(f"pages({hp})")

    meta = {"source": "+".join(sources) if sources else "none",
            "count_found": len(acc),
            "board_pin_total": board_pin_total}
    return acc, meta

# ===============================
# App UI
# ===============================

st.set_page_config(page_title="Pinterest Board Color Analyzer", layout="wide")

# Title (no link icon) + credit
st.markdown("<h1 style='margin:0 0 0.25rem 0;font-weight:700;'>Pinterest Board Color Analyzer</h1>", unsafe_allow_html=True)
st.caption("By: [Wui-Liang Lim](https://www.linkedin.com/in/limwuiliang/)")
st.caption("Paste a **public** Pinterest board URL and click Analyze.")

board_url = st.text_input("Pinterest board URL", placeholder="https://www.pinterest.com/<username>/<board-slug>/")

with st.expander("Settings", expanded=False):
    pin_limit = st.slider("Max pins to analyze", 30, 500, 120, step=10)
    palette_k = st.slider("Colors per image (KMeans)", 3, 8, 5)
    master_palette_k = st.slider("Master palette size (across board)", 5, 20, 10)
    thumb_size = st.slider("Pin thumbnail size (px)", 90, 200, 120, step=10)

with st.expander("Advanced (optional)", expanded=False):
    st.markdown("**Use a cookie to go beyond the ~50-pin public cap (if your board has more).**")
    with st.expander("How to get your Pinterest cookie (Chrome)", expanded=False):
        st.markdown("""
1. Open **pinterest.com** and sign in.  
2. Press **F12** (Windows/Linux) or **⌥⌘I** (Mac) → open **Network** tab.  
3. Refresh. Click any request to `www.pinterest.com`.  
4. In **Request Headers**, copy the entire **cookie:** line (e.g. `sb=…; _pinterest_sess=…; csrftoken=…; …`).  
5. Paste it below. *Your cookie is used only for this run and not stored.*  
        """)
    cookie_text = st.text_input("Cookie header", value="", type="password",
                                placeholder="sb=...; _pinterest_sess=...; csrftoken=...; ...")
    cookie_ok = bool(re.search(r"_pinterest_sess=.*csrftoken=", cookie_text))
    st.caption(("✅ Cookie looks OK (found `_pinterest_sess` & `csrftoken`)."
                if cookie_ok else "ℹ️ Optional. Add `_pinterest_sess` + `csrftoken` to fetch more than ~50 pins."))
    cookie = cookie_text or None

if not st.button("Analyze"):
    st.stop()

if not board_url or "pinterest." not in urlparse(board_url).netloc:
    st.error("Please paste a valid public Pinterest board URL."); st.stop()

# ---- Scrape ----
pins, meta = scrape_board_boardonly(board_url, cookie=cookie, max_pins=pin_limit*2)
if not pins:
    st.error("No pins found. The board may be private or region-blocked."); st.stop()

if (not cookie) and meta.get("board_pin_total") and meta["board_pin_total"] > len(pins):
    st.info(
        f"This board reports **{meta['board_pin_total']}** pins. Public endpoints often cap near **50** without a cookie. "
        "Add a cookie in **Advanced** to fetch the remainder."
    )

pins_df = pd.DataFrame(pins[:pin_limit])

# ===============================
# Color extraction (+ inline thumbnails for Explorer)
# ===============================

progress = st.progress(0); records, fetch_failures = [], 0
for idx, row in pins_df.iterrows():
    arr = load_image_as_array(row["image_url"], cookie=cookie)
    if arr is None:
        fetch_failures += 1; continue
    pixels = arr.reshape(-1,3)
    if pixels.shape[0] > 6000:
        sel = np.random.RandomState(42).choice(pixels.shape[0], 6000, replace=False)
        pixels = pixels[sel]
    km = KMeans(n_clusters=palette_k, n_init="auto", random_state=42).fit(pixels)
    centers = km.cluster_centers_.astype(float)
    hsv = rgb_to_hsv_np(centers)
    hexes = [hex_from_rgb(c) for c in centers]
    thumb_uri = array_to_data_uri(arr, max_side=112, fmt="JPEG", quality=78)
    records.append({
        "pin_id": row.get("pin_id"),
        "image_url": row["image_url"],
        "thumb_uri": thumb_uri,
        "title": row.get("title",""),
        "description": row.get("description",""),
        "created_at": row.get("created_at"),
        "palette_hex": hexes, "palette_rgb": centers.tolist(),
        "palette_hsv": hsv.tolist(), "dominant_hex": hexes[0] if hexes else None,
    })
    progress.progress(min(1.0, (len(records)/len(pins_df))))
progress.empty()
if not records:
    st.error("Images could not be processed."); st.stop()
colors_df = pd.DataFrame(records)

# Long form + dominant rows
pal_rows, dom_rows = [], []
for _, r in colors_df.iterrows():
    for i, hx in enumerate(r["palette_hex"]):
        rgb = r["palette_rgb"][i]; hsv = r["palette_hsv"][i]
        pal_rows.append({"pin_id": r["pin_id"], "image_url": r["image_url"], "title": r["title"],
                         "hex": hx, "r": rgb[0], "g": rgb[1], "b": rgb[2], "h": hsv[0], "s": hsv[1], "v": hsv[2]})
    if r["palette_hsv"]:
        dh, _, dv = r["palette_hsv"][0]
        dom_rows.append({"pin_id": r["pin_id"], "image_url": r["image_url"], "thumb": r["thumb_uri"], "title": r["title"],
                         "h_deg": float(dh)*360.0, "v": float(dv), "hex": r["palette_hex"][0]})
pal_df = pd.DataFrame(pal_rows)
dom_df = pd.DataFrame(dom_rows)

# ===============================
# Master Palette (DESC)
# ===============================

st.subheader("Master Color Palette")

all_rgb = pal_df[["r","g","b"]].to_numpy()
master = KMeans(n_clusters=master_palette_k, n_init="auto", random_state=42).fit(all_rgb)
centers = master.cluster_centers_.astype(float)
master_hex = [hex_from_rgb(c) for c in centers]

pal_df["cluster"] = master.predict(all_rgb)
counts = pal_df["cluster"].value_counts()
order_idx = list(counts.sort_values(ascending=False).index)
rank_map = {cid: rank for rank, cid in enumerate(order_idx)}
master_hex_sorted = [master_hex[cid] for cid in order_idx]
counts_sorted = [int(counts.get(cid, 0)) for cid in order_idx]

cols = st.columns(min(6, master_palette_k))
for i, hx in enumerate(master_hex_sorted):
    with cols[i % len(cols)]:
        st.markdown(f"**C{i+1} — {hx}**")
        st.markdown(
            f"<div style='width:100%;height:42px;border-radius:8px;border:1px solid #ddd;background:{hx};'></div>"
            f"<div style='font-size:12px;color:#666;'>share: {counts_sorted[i]}</div>",
            unsafe_allow_html=True,
        )

share_df = pd.DataFrame({"cluster_label":[f"C{i+1}" for i in range(len(master_hex_sorted))],
                         "count":counts_sorted, "hex":master_hex_sorted})
# Sorted descending so labels match palette order
st.altair_chart(
    alt.Chart(share_df).mark_bar(stroke="black", strokeWidth=0.25).encode(
        x=alt.X("cluster_label:N", sort='-y', title="Cluster (DESC)"),
        y=alt.Y("count:Q", title="Frequency"),
        color=alt.Color("hex:N", scale=None, legend=None),
        tooltip=["cluster_label","hex","count"],
    ).properties(height=240),
    use_container_width=True
)

# ===============================
# Pin Gallery (sorted by C1..)
# ===============================

def pin_cluster_rank(row):
    if not row["palette_rgb"]: return 10**9
    dom_rgb = np.array(row["palette_rgb"][0]).reshape(1, -1)
    cid = int(master.predict(dom_rgb)[0])
    return rank_map.get(cid, 10**9)

colors_df["cluster_rank"] = colors_df.apply(pin_cluster_rank, axis=1)

st.subheader("Pin Gallery")
st.caption(f"Showing {len(colors_df)} of {len(pins_df)} pins (image fetch failures: {fetch_failures}) — sorted by C1, C2, …")

st.markdown(f"""
<style>
.pin-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax({thumb_size}px,1fr)); gap:8px; }}
.pin-card {{ position:relative; aspect-ratio:1/1; overflow:hidden; border-radius:8px; border:1px solid #ddd; background:#f7f7f7; }}
.pin-card img {{ width:100%; height:100%; object-fit:cover; display:block; }}
.pin-overlay {{ position:absolute; left:0; right:0; bottom:0; background:rgba(255,255,255,.96);
  transform:translateY(100%); transition:transform .16s ease; padding:6px 8px; border-top:1px solid #eee; }}
.pin-card:hover .pin-overlay {{ transform:translateY(0%); }}
.palette-row {{ display:grid; grid-template-columns:repeat(5,1fr); gap:4px; margin-top:4px; }}
.swatch {{ height:12px; border-radius:4px; border:1px solid rgba(0,0,0,.1); }}
.hexline {{ margin-top:4px; font-size:11px; color:#333; line-height:1.2; word-break:break-all; }}
.title {{ font-size:12px; color:#222; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
</style>
""", unsafe_allow_html=True)

cards = []
for _, r in colors_df.sort_values(["cluster_rank","title"]).iterrows():
    t = html.escape((r.get("title") or "").strip())
    palette = r.get("palette_hex") or []
    swatches = "".join([f"<div class='swatch' style='background:{html.escape(hx)}'></div>" for hx in palette[:5]])
    hexline = " ".join([html.escape(hx) for hx in palette[:5]])
    img_url = html.escape(r.get("image_url") or "")
    cards.append(f"""
<div class='pin-card'>
  <img src='{img_url}' loading='lazy' alt='{t}'>
  <div class='pin-overlay'>
    <div class='title'>{t}</div>
    <div class='palette-row'>{swatches}</div>
    <div class='hexline'>{hexline}</div>
  </div>
</div>""")
st.markdown(f"<div class='pin-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)

# ===============================
# Hue Histogram + Radial Map
# ===============================

st.subheader("Hue Distribution")
pal_df["h_deg"] = (pal_df["h"] * 360.0).round(1)
bins = np.arange(0, 361, 10); labels = (bins[:-1] + bins[1:]) / 2
pal_df["h_bin"] = pd.cut(pal_df["h_deg"], bins=bins, include_lowest=True, labels=labels)
h_bin_df = pal_df.groupby("h_bin").size().reset_index(name="count")
h_bin_df["h_mid"] = h_bin_df["h_bin"].astype(float)
h_bin_df["h_color"] = h_bin_df["h_mid"].apply(lambda d: f"hsl({int(d)},90%,50%)")
st.altair_chart(
    alt.Chart(h_bin_df).mark_bar(stroke="black", strokeWidth=0.25).encode(
        x=alt.X("h_mid:Q", title="Hue (°)"),
        y=alt.Y("count:Q", title="Count"),
        color=alt.Color("h_color:N", scale=None, legend=None),
        tooltip=["h_mid","count"],
    ).properties(height=240),
    use_container_width=True
)

st.subheader("Hue × Value Radial Map (Smoothed)")
hv_h = pal_df["h"].to_numpy()*360.0
hv_v = pal_df["v"].to_numpy()
rgb_arr = pal_df[["r","g","b"]].to_numpy()
H_STEPS, V_STEPS, SIGMA_H, SIGMA_V, OUTER_R, INNER_PAD = 72, 24, 18.0, 0.12, 140, 24

def smooth_color(hc, vc):
    dh = np.abs(hc - hv_h); dh = np.minimum(dh, 360.0 - dh)
    dv = np.abs(vc - hv_v)
    w = np.exp(-(dh**2)/(2*SIGMA_H**2) - (dv**2)/(2*SIGMA_V**2))
    ws = w.sum()
    if ws <= 1e-9: return None
    rgb = (rgb_arr * w[:,None]).sum(axis=0) / ws
    return hex_from_rgb(rgb)

theta_span = 360.0 / H_STEPS
v_edges = np.linspace(0.0, 1.0, V_STEPS+1)
v_centers = (v_edges[:-1] + v_edges[1:]) / 2
h_centers = np.linspace(0.0, 360.0 - theta_span, H_STEPS)
radial_rows = []
for v_c, v_lo, v_hi in zip(v_centers, v_edges[:-1], v_edges[1:]):
    inner = INNER_PAD + v_lo * OUTER_R
    outer = INNER_PAD + v_hi * OUTER_R
    for h_c in h_centers:
        hx = smooth_color(h_c, v_c)
        if not hx: continue
        radial_rows.append({"theta": h_c - theta_span/2, "theta2": h_c + theta_span/2,
                            "radius": inner, "radius2": outer, "hex": hx,
                            "h_center": h_c, "v_center": float(v_c)})
radial_df = pd.DataFrame(radial_rows)
st.altair_chart(
    alt.Chart(radial_df).mark_arc(stroke=None).encode(
        theta="theta:Q", theta2="theta2:Q",
        radius="radius:Q", radius2="radius2:Q",
        color=alt.Color("hex:N", scale=None, legend=None),
        tooltip=["h_center","v_center"],
    ).properties(width=380, height=380),
    use_container_width=False
)

# ===============================
# Hue × Value Explorer
# ===============================

st.subheader("Hue × Value Explorer (drag to filter & see pins)")

def render_thumb_grid(df, cols=14, size=56):
    # data-URI thumbnails stacked as an HTML grid (tiny gaps)
    n = len(df)
    rows = max(1, math.ceil(n/cols))
    cells = []
    for _, r in df.iterrows():
        uri = html.escape(r.get("thumb") or r.get("thumb_uri") or "")
        title = html.escape(r.get("title") or "")
        cells.append(f"<img src='{uri}' alt='{title}' width='{size}' height='{size}' style='object-fit:cover;border-radius:6px;border:1px solid #ddd;'>")
    st.markdown(
        f"<div style='display:grid;grid-template-columns:repeat({cols},{size}px);gap:2px;'>{''.join(cells)}</div>",
        unsafe_allow_html=True
    )

if not dom_df.empty:
    # Base scatter + selection
    brush = alt.selection_interval(encodings=["x","y"])
    scatter = (
        alt.Chart(dom_df)
        .mark_circle(size=60, opacity=0.9)
        .encode(
            x=alt.X("h_deg:Q", title="Hue (°)", scale=alt.Scale(domain=[0,360])),
            y=alt.Y("v:Q", title="Value (0–1)", scale=alt.Scale(domain=[0,1])),
            color=alt.Color("hex:N", scale=None, legend=None),
            tooltip=["title:N","hex:N","h_deg:Q","v:Q"],
        )
        .add_params(brush)
        .properties(height=240)
    )

    # Thumbnails chart (same Vega spec) — may fail on some Altair/vega-lite combos
    thumb_e, grid_cols = 56, 14
    thumbs_spec = (
        alt.Chart(dom_df)
        .transform_filter(brush)
        .transform_window(rn="row_number()")
        .transform_calculate(col=f"datum.rn % {grid_cols}", row=f"floor(datum.rn / {grid_cols})")
        .mark_image(width=thumb_e, height=thumb_e)
        .encode(
            x=alt.X("col:O", axis=None, sort=None, scale=alt.Scale(padding=0, paddingInner=0, paddingOuter=0)),
            y=alt.Y("row:O", axis=None, sort=None, scale=alt.Scale(padding=0, paddingInner=0, paddingOuter=0)),
            url="thumb:N",
            tooltip=["title:N"],
        )
        .properties(width=grid_cols*thumb_e, height=thumb_e * (max(1, math.ceil(len(dom_df)/grid_cols))))
        .configure_scale(bandPaddingInner=0, bandPaddingOuter=0)
        .configure_view(strokeWidth=0)
    )

    # Try linked-spec first; if Altair rejects concat/vconcat, fall back to sliders + HTML grid
    try:
        explorer = alt.vconcat(scatter, thumbs_spec)  # one Vega spec so brush works
        st.altair_chart(explorer, use_container_width=True)
    except Exception:
        st.warning("Interactive brush linking isn’t available in this environment. Using slider filter instead.")
        st.altair_chart(scatter.properties(title=None), use_container_width=True)
        # Fallback sliders drive the thumbnail grid
        c1, c2 = st.columns(2)
        with c1:
            hue_range = st.slider("Hue range (°)", 0, 360, (0, 360), step=1)
        with c2:
            val_range = st.slider("Value range", 0.0, 1.0, (0.0, 1.0), step=0.01)
        filtered = dom_df[(dom_df["h_deg"].between(hue_range[0], hue_range[1])) &
                          (dom_df["v"].between(val_range[0], val_range[1]))]
        render_thumb_grid(filtered.rename(columns={"thumb":"thumb_uri"}), cols=14, size=56)
else:
    st.info("No dominant-color points available for the explorer.")

# ===============================
# Diagnostics
# ===============================

with st.expander("🔧 Diagnostics"):
    st.write({
        "board_url": board_url,
        "pins_parsed_total": int(len(pins_df)),
        "source_union": meta.get("source"),
        "count_kept_after_dedupe": meta.get("count_found"),
        "board_pin_total": meta.get("board_pin_total"),
        "image_fetch_failures": fetch_failures,
    })
