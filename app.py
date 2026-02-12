from flask import Flask, render_template, request, jsonify
import os
import uuid
import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
import io

# =============================
# App / Folders
# =============================
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OVERLAY_FOLDER = "static/overlays"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OVERLAY_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_EXT


# -----------------------------
# CLIP image normalize
# -----------------------------
CLIP_MAX_SIDE = 768
CLIP_JPEG_QUALITY = 85
CLIP_MAX_BYTES = 900_000


def clip_normalize_to_jpeg_bytes(path: str) -> bytes:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")

    w, h = img.size
    scale = min(1.0, CLIP_MAX_SIDE / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=CLIP_JPEG_QUALITY, optimize=True)
    data = buf.getvalue()

    if len(data) > CLIP_MAX_BYTES:
        q = 75
        while q >= 55:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q, optimize=True)
            data = buf.getvalue()
            if len(data) <= CLIP_MAX_BYTES:
                break
            q -= 10

    return data


def clamp01(x: float) -> float:
    x = float(x)
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return x


# =============================
# Utils (non-CLIP scoring)
# =============================
def corr_to_01(c):
    return clamp01((float(c) + 1.0) / 2.0)


def diff_to_01(d, maxv):
    return clamp01(1.0 - (float(d) / float(maxv)))


def get_histogram(path):
    img = cv2.imread(path)
    if img is None:
        print("⚠️ read failed:", path)
        return None

    img = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.astype("float32")


def orb_shape_score(path1, path2):
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        return 0.0

    orb = cv2.ORB_create(nfeatures=800)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = 0
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good += 1

    denom = max(1, min(len(kp1), len(kp2)))
    return clamp01(good / denom)


def make_overlay_with_bbox(base_path, target_path):
    base = cv2.imread(base_path)
    tgt = cv2.imread(target_path)
    if base is None or tgt is None:
        return None, None

    h, w = tgt.shape[:2]
    base_rs = cv2.resize(base, (w, h))

    diff = cv2.absdiff(base_rs, tgt)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((7, 7), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.dilate(th, kernel, iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = tgt.copy()

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
        bbox = (int(x), int(y), int(bw), int(bh))
    else:
        bbox = None

    outname = f"{uuid.uuid4()}.png"
    outpath = os.path.join(OVERLAY_FOLDER, outname)
    cv2.imwrite(outpath, overlay)
    return outname, bbox


def get_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def get_edge_density(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return float(np.mean(edges > 0))


# =============================
# Routes
# =============================
HF_BATCH_URL = "https://taku1103-clip-sim-api.hf.space/clip_sim_batch"
TOP_K = 30
W_OPENCV = 0.7
W_CLIP = 0.3


def opencv_to_0_100(hist_score: float) -> int:
    # compareHist(CORREL) は -1..1 → 0..1 → 0..100
    s01 = clamp01((float(hist_score) + 1.0) / 2.0)
    return int(round(s01 * 100))


def parse_min_score100(val: str, default100: int = 40) -> int:
    try:
        x = float(val)
    except Exception:
        return int(default100)

    # 0..100 以外は丸めるだけ（0..1互換を捨てる）
    return int(max(0, min(100, round(x))))


@app.route("/search_api", methods=["POST"])
def search_api():
    """
    JSからFormDataで呼ぶ：
      - base: file (optional)
      - prev_base: str (optional)
      - folder: files (multiple)
      - threshold: str (0..100 推奨 / 0..1 も互換)
      - filter_mode: "final" or "opencv"

    return JSON:
      { ok, base, threshold100, mode, results, results_all }
    """
    # 1) params
    prev_base = request.form.get("prev_base") or ""
    prev_base = os.path.basename(prev_base) if prev_base else ""

    mode = request.form.get("filter_mode", "final")
    if mode not in ("final", "opencv"):
        mode = "final"

    min_score100 = parse_min_score100(request.form.get("min_score", "40"), default100=40)

    base_file = request.files.get("base")
    folder_files = request.files.getlist("folder")

    # 2) base決定
    if base_file and base_file.filename:
        if not allowed_file(base_file.filename):
            return jsonify({"ok": False, "error": "対応していない画像形式です😢"}), 400
        ext = os.path.splitext(base_file.filename)[1].lower()
        base_name = str(uuid.uuid4()) + ext
        base_path = os.path.join(app.config["UPLOAD_FOLDER"], base_name)
        base_file.save(base_path)
    elif prev_base:
        base_name = prev_base
        base_path = os.path.join(app.config["UPLOAD_FOLDER"], base_name)
        if not os.path.exists(base_path):
            return jsonify({"ok": False, "error": "基準画像をもう一度選んでね📸"}), 400
    else:
        return jsonify({"ok": False, "error": "基準画像を選んでね📸"}), 400

    base_hist = get_histogram(base_path)
    if base_hist is None:
        return jsonify({"ok": False, "error": "基準画像が壊れています😢"}), 400

    # 3) OpenCVスコア（ここでは *絞らない*：全件保持）
    candidates = []  # [{"name","hist","hist100","path"}]
    for f in folder_files:
        if not f or not f.filename:
            continue
        if not allowed_file(f.filename):
            continue

        ext = os.path.splitext(f.filename)[1].lower()
        name = str(uuid.uuid4()) + ext
        path = os.path.join(app.config["UPLOAD_FOLDER"], name)
        f.save(path)

        hist = get_histogram(path)
        if hist is None:
            continue

        score = float(cv2.compareHist(base_hist, hist, cv2.HISTCMP_CORREL))
        hist100 = opencv_to_0_100(score)
        candidates.append({"name": name, "hist": score, "hist100": hist100, "path": path})

    if not candidates:
        return jsonify({
            "ok": True,
            "base": base_name,
            "threshold100": min_score100,
            "mode": mode,
            "results": [],
            "results_all": []
        })

    # 4) CLIPは OpenCV上位 TOP_K だけに投げる
    candidates.sort(key=lambda x: x["hist100"], reverse=True)
    top = candidates[:TOP_K]

    clip_map = {c["name"]: 0 for c in top}
    try:
        files = []
        files.append(("base", ("base.jpg", clip_normalize_to_jpeg_bytes(base_path), "image/jpeg")))
        for c in top:
            files.append(("targets", (c["name"], clip_normalize_to_jpeg_bytes(c["path"]), "image/jpeg")))

        resp = requests.post(HF_BATCH_URL, files=files, timeout=75)
        data = resp.json() if resp.ok else None
        if not data or not data.get("ok"):
            raise RuntimeError(
                data.get("error") if isinstance(data, dict) else f"HF error HTTP {resp.status_code}"
            )

        for r in data.get("results", []):
            clip_map[r.get("name")] = int(r.get("clip", 0))
    except Exception as e:
        print("HF batch failed:", e)

    # 5) 合成：全件 out_all
    out_all = []
    for c in candidates:
        clip100 = int(clip_map.get(c["name"], 0))  # TOP_K外は0
        final100 = int(round(W_OPENCV * int(c["hist100"]) + W_CLIP * clip100))
        out_all.append({
            "name": c["name"],
            "hist": c["hist"],
            "hist100": int(c["hist100"]),
            "clip": clip100,
            "final100": final100
        })

    # 6) mode + min_score100 で絞って返す
    if mode == "opencv":
        out_filtered = [r for r in out_all if int(r["hist100"]) >= min_score100]
        out_filtered.sort(key=lambda x: x["hist100"], reverse=True)
    else:
        out_filtered = [r for r in out_all if int(r["final100"]) >= min_score100]
        out_filtered.sort(key=lambda x: x["final100"], reverse=True)

    # out_all は基本 final100順（見やすさ）
    out_all.sort(key=lambda x: x["final100"], reverse=True)

    return jsonify({
        "ok": True,
        "base": base_name,
        "threshold100": min_score100,
        "mode": mode,
        "results": out_filtered,
        "results_all": out_all
    })


@app.route("/", methods=["GET"])
def index():
    # 初期表示はJS検索前提なので、ここは薄くでOK
    return render_template("index.html", base=None, results=None, threshold=40)


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    ここは軽量分析のみ（CLIPは叩かない）
    """
    data = request.json or {}
    base = data.get("base")
    target = data.get("target")
    if not base or not target:
        return jsonify({"text": "パラメータが不足しています😢"})

    base_path = os.path.join(app.config["UPLOAD_FOLDER"], os.path.basename(base))
    target_path = os.path.join(app.config["UPLOAD_FOLDER"], os.path.basename(target))

    if not os.path.exists(base_path) or not os.path.exists(target_path):
        return jsonify({"text": "解析対象が見つかりませんでした😢"})

    bimg = cv2.imread(base_path)
    timg = cv2.imread(target_path)
    if bimg is None or timg is None:
        return jsonify({"text": "画像を読み込めませんでした😢"})

    # 色
    bh = get_histogram(base_path)
    th = get_histogram(target_path)
    color_sim = 0.0
    if bh is not None and th is not None:
        color_sim = corr_to_01(cv2.compareHist(bh, th, cv2.HISTCMP_CORREL))

    # 明るさ
    b1, b2 = get_brightness(bimg), get_brightness(timg)
    bright_sim = diff_to_01(abs(b1 - b2), 80.0)

    # 構造
    e1, e2 = get_edge_density(bimg), get_edge_density(timg)
    edge_sim = diff_to_01(abs(e1 - e2), 0.15)

    # ORB
    orb_sim = orb_shape_score(base_path, target_path)

    # overlay
    overlay_name, bbox = make_overlay_with_bbox(base_path, target_path)
    overlay_url = f"/static/overlays/{overlay_name}" if overlay_name else f"/static/uploads/{target}"

    # 総合（CLIPなし）
    overall = (0.30 * color_sim + 0.20 * bright_sim + 0.20 * edge_sim + 0.30 * orb_sim)
    score100 = int(round(clamp01(overall) * 100))

    reasons = []
    if color_sim > 0.75:
        reasons.append("色合いがかなり近い")
    elif color_sim > 0.55:
        reasons.append("色合いがそこそこ似ている")

    if bright_sim > 0.75:
        reasons.append("明るさが近い")
    if edge_sim > 0.75:
        reasons.append("輪郭の情報量（構造）が近い")
    if orb_sim > 0.20:
        reasons.append("形の一致（特徴点）が多い")
    if not reasons:
        reasons.append("全体の特徴が近い可能性がある")

    text = (
        f"総合 {score100}点。"
        f"主な理由：{'・'.join(reasons)}。"
        f"（色:{int(color_sim*100)} 明:{int(bright_sim*100)} 構:{int(edge_sim*100)} 形:{int(orb_sim*100)}）"
    )

    return jsonify({
        "text": text,
        "score100": score100,
        "metrics": {
            "color": int(color_sim * 100),
            "brightness": int(bright_sim * 100),
            "structure": int(edge_sim * 100),
            "shape": int(orb_sim * 100),
        },
        "overlay_url": overlay_url,
        "bbox": bbox
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)