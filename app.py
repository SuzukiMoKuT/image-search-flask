from flask import Flask, render_template, request, jsonify
import os
import uuid
import cv2
import numpy as np

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
def corr_to_01(c):  # -1..1 -> 0..1
    return clamp01((float(c) + 1.0) / 2.0)


def diff_to_01(d, maxv):  # 差が小さいほど1に近い
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
@app.route("/", methods=["GET", "POST"])
def index():
    threshold = 0.4
    base_name = None
    results = []

    if request.method == "POST":
        prev_base = request.form.get("prev_base") or ""
        prev_base = os.path.basename(prev_base) if prev_base else ""

        base = request.files.get("base")
        files = request.files.getlist("folder")

        try:
            threshold = float(request.form.get("threshold", 0.4))
        except Exception:
            threshold = 0.4

        # base決定
        if base and base.filename:
            if not allowed_file(base.filename):
                return render_template("index.html", error="対応していない画像形式です😢")
            ext = os.path.splitext(base.filename)[1].lower()
            base_name = str(uuid.uuid4()) + ext
            base_path = os.path.join(app.config["UPLOAD_FOLDER"], base_name)
            base.save(base_path)
        elif prev_base:
            base_name = prev_base
            base_path = os.path.join(app.config["UPLOAD_FOLDER"], base_name)
            if not os.path.exists(base_path):
                return render_template("index.html", error="基準画像をもう一度選んでね📸")
        else:
            return render_template("index.html", error="基準画像を選んでね📸")

        base_hist = get_histogram(base_path)
        if base_hist is None:
            return render_template("index.html", error="基準画像が壊れています😢")

        # folder内を比較（ヒストグラム相関）
        for f in files:
            if not f or not f.filename:
                continue
            if not allowed_file(f.filename):
                continue

            ext = os.path.splitext(f.filename)[1].lower()
            new_name = str(uuid.uuid4()) + ext
            path = os.path.join(app.config["UPLOAD_FOLDER"], new_name)
            f.save(path)

            hist = get_histogram(path)
            if hist is None:
                continue

            score = cv2.compareHist(base_hist, hist, cv2.HISTCMP_CORREL)
            if score >= threshold:
                results.append((new_name, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)

    return render_template("index.html", base=base_name, results=results, threshold=threshold)


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    ここは軽量分析のみ（CLIPは絶対に叩かない）
    """
    data = request.json or {}
    base = data.get("base")
    target = data.get("target")
    if not base or not target:
        return jsonify({"text": "パラメータが不足しています😢"})

    base_path = os.path.join(app.config["UPLOAD_FOLDER"], os.path.basename(base))
    target_path = os.path.join(app.config["UPLOAD_FOLDER"], os.path.basename(target))

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
    # debug=True は開発だけで。Render本番は gunicorn 推奨
    app.run(host="0.0.0.0", port=5000, debug=True)