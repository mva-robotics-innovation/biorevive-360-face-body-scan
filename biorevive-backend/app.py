import base64, os
import cv2, numpy as np, mediapipe as mp
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__)); MD = os.path.join(HERE, "models")
app = Flask(__name__); CORS(app)
try:
    from download_models import ensure
    ensure()
except Exception as e:
    print("model download skipped:", e, flush=True)

# ---- Models: MediaPipe landmarks (478 pts) + InsightFace genderage (ONNX, age regression) ----
FL = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=mpt.BaseOptions(model_asset_path=os.path.join(MD, "face_landmarker.task")),
    running_mode=vision.RunningMode.IMAGE, num_faces=1))
GA_PATH = os.path.join(MD, "genderage.onnx"); ORT = NET = None
if os.path.exists(GA_PATH):
    try:
        import onnxruntime as ort
        ORT = ort.InferenceSession(GA_PATH, providers=["CPUExecutionProvider"])
    except Exception:
        try: NET = cv2.dnn.readNetFromONNX(GA_PATH)
        except Exception as e: print("genderage load failed:", e, flush=True)
HAS_DL = ORT is not None or NET is not None
print("Deep age/gender model:", "ONNX Runtime" if ORT else "OpenCV DNN" if NET else "NOT LOADED", flush=True)

def open_model(name):
    p = os.path.join(MD, name)
    if not os.path.exists(p): return None
    try:
        import onnxruntime as ort
        return ("ort", ort.InferenceSession(p, providers=["CPUExecutionProvider"]))
    except Exception:
        try: return ("cv", cv2.dnn.readNetFromONNX(p))
        except Exception as e: print(name, "load failed:", e, flush=True); return None

def run_model(m, x):
    if m[0] == "ort": return [np.asarray(o) for o in m[1].run(None, {m[1].get_inputs()[0].name: x})]
    m[1].setInput(x); o = m[1].forward(m[1].getUnconnectedOutLayersNames())
    return [np.asarray(v) for v in (o if isinstance(o, (list, tuple)) else [o])]

FF, EM = open_model("fairface.onnx"), open_model("emotion-ferplus-8.onnx")
print("FairFace age model:", bool(FF), "| FER+ expression model:", bool(EM), flush=True)
ARC = np.array([[.34191607, .46157411], [.65653393, .45983393], [.500225, .64050536], [.37097589, .82469196], [.63151696, .82325089]], np.float32) * 224
FF_MIDS = np.array([1, 6, 15, 25, 35, 45, 55, 65, 75], np.float32)   # FairFace buckets: 0-2,3-9,10-19,...,60-69,70+
EMO = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

def fairface_age(img, P):
    """Expected age from FairFace's 9-bucket softmax (output picked by size, so no gender/race is used)."""
    src = np.array([(P[33] + P[133]) / 2, (P[362] + P[263]) / 2, P[1], P[61], P[291]], np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, ARC, method=cv2.LMEDS)
    if M is None: return []
    crop = cv2.warpAffine(img, M, (224, 224), borderMode=cv2.BORDER_REPLICATE); out = []
    for c in (crop, crop[:, ::-1]):
        x = ((c[:, :, ::-1] / 255.0 - [.485, .456, .406]) / [.229, .224, .225]).transpose(2, 0, 1)[None].astype(np.float32)
        for o in run_model(FF, x):
            o = o.reshape(-1)
            if o.size == 9: e = np.exp(o - o.max()); out.append(float((e / e.sum()) @ FF_MIDS))
    return out

def expression(face_gray):
    o = run_model(EM, cv2.resize(face_gray, (64, 64)).astype(np.float32)[None, None])[0].reshape(-1)
    e = np.exp(o - o.max()); p = e / e.sum(); i = int(p.argmax()); return EMO[i], float(p[i])

OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
LIPS = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]
LEYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
REYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
clip = lambda v, a=0, b=100: float(np.clip(v, a, b))

def ga_forward(blob):
    if ORT is not None: return ORT.run(None, {ORT.get_inputs()[0].name: blob})[0][0]
    NET.setInput(blob); return NET.forward()[0]

def deep_age_gender(img, P):
    """Test-time augmentation: 3 crop scales x horizontal flip. Returns (ages[], male_probs[])."""
    x0, y0 = P.min(0); x1, y1 = P.max(0); c = ((x0 + x1) / 2, (y0 + y1) / 2); side = max(x1 - x0, y1 - y0)
    ages, males = [], []
    for k in (1.4, 1.5, 1.6):
        s = 96 / (side * k)
        M = np.array([[s, 0, 48 - c[0] * s], [0, s, 48 - c[1] * s]], np.float32)
        a = cv2.warpAffine(img, M, (96, 96), borderValue=0)
        for crop in (a, a[:, ::-1]):
            p = ga_forward(cv2.dnn.blobFromImage(np.ascontiguousarray(crop), 1.0, (96, 96), (0, 0, 0), swapRB=True))
            e = np.exp(p[:2] - p[:2].max()); males.append(float(e[1] / e.sum())); ages.append(float(p[2] * 100))
    return ages, males

def analyze(img):
    h, w = img.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    res = FL.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks: return None
    P = np.array([[l.x * w, l.y * h] for l in res.face_landmarks[0]])
    fw = abs(P[454, 0] - P[234, 0]); ew = abs(P[133, 0] - P[33, 0])
    yaw = (P[1, 0] - (P[234, 0] + P[454, 0]) / 2) / (fw + 1e-6)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    def roi(a, xa, xb, ya, yb):
        xa, xb = sorted((int(max(0, xa)), int(min(w, xb)))); ya, yb = sorted((int(max(0, ya)), int(min(h, yb))))
        return a[ya:yb, xa:xb]
    def edge(a):
        return 0.0 if a.size < 100 else float((cv2.Canny(cv2.GaussianBlur(a, (3, 3), 0), 30, 90) > 0).mean() * 100)

    fore = edge(roi(gray, P[107, 0], P[336, 0], P[10, 1], P[105, 1] - ew * .15))
    crow = np.mean([edge(roi(gray, P[33, 0] - ew * .6, P[33, 0] - ew * .05, P[33, 1] - ew * .2, P[33, 1] + ew * .5)),
                    edge(roi(gray, P[263, 0] + ew * .05, P[263, 0] + ew * .6, P[263, 1] - ew * .2, P[263, 1] + ew * .5))])
    nasol = np.mean([edge(roi(gray, P[205, 0] - ew * .3, P[61, 0], P[205, 1] - ew * .2, P[61, 1] + ew * .3)),
                     edge(roi(gray, P[291, 0], P[425, 0] + ew * .3, P[425, 1] - ew * .2, P[291, 1] + ew * .3))])
    cr = [roi(a, P[i, 0] - ew * .35, P[i, 0] + ew * .35, P[i, 1] - ew * .25, P[i, 1] + ew * .25) for i in (50, 280) for a in (gray, lab)]
    cg, cl = [cr[0], cr[2]], [cr[1], cr[3]]
    lap = np.mean([cv2.Laplacian(cv2.GaussianBlur(c, (3, 3), 0), cv2.CV_64F).var() for c in cg if c.size] or [0])
    pig = np.mean([cv2.GaussianBlur(c[:, :, 0], (7, 7), 0).std() for c in cl if c.size] or [0])
    red = np.mean([c[:, :, 1].mean() - 128 for c in cl if c.size] or [0])
    ue = [roi(lab, P[33, 0], P[133, 0], P[145, 1] + ew * .05, P[145, 1] + ew * .35),
          roi(lab, P[362, 0], P[263, 0], P[374, 1] + ew * .05, P[374, 1] + ew * .35)]
    cheekL = np.mean([c[:, :, 0].mean() for c in cl if c.size] or [0])
    dark = np.mean([max(0, cheekL - u[:, :, 0].mean()) for u in ue if u.size] or [0])

    # skin mask (face oval minus eyes/lips, YCrCb skin range) -> spots + shine
    mask = np.zeros((h, w), np.uint8); cv2.fillPoly(mask, [P[OVAL].astype(np.int32)], 255)
    for idx in (LIPS, LEYE, REYE): cv2.fillPoly(mask, [P[idx].astype(np.int32)], 0)
    mask = cv2.erode(cv2.bitwise_and(mask, cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb), (0, 133, 77), (255, 173, 127))), np.ones((5, 5), np.uint8))
    n = int((mask > 0).sum()); L = lab[:, :, 0]
    spots = float(((L - cv2.GaussianBlur(L, (0, 0), 12) < -9) & (mask > 0)).sum() / n * 100) if n > 500 else 0.0
    shine = float(((L > 235) & (mask > 0)).sum() / n * 100) if n > 500 else 0.0

    ax = P[152] - P[10]; nrm = np.array([-ax[1], ax[0]]) / (np.linalg.norm(ax) + 1e-6)
    pairs = [(33, 263), (61, 291), (234, 454), (70, 300), (129, 358), (172, 397)]
    asym = np.mean([abs((P[a] - P[10]) @ nrm + (P[b] - P[10]) @ nrm) for a, b in pairs]) / (fw + 1e-6)

    s = dict(wrinkles=clip((fore * .4 + crow * .35 + nasol * .25) * 8), texture=clip(np.log1p(lap) * 16 - 35),
             dark_circles=clip(dark * 3.3), pigmentation=clip(pig * 8), redness=clip(red * 4),
             symmetry=clip(100 - asym * 600), spots=clip(spots * 10), shine=clip(shine * 400))
    s["aging_index"] = clip(.30 * s["wrinkles"] + .18 * s["texture"] + .17 * s["dark_circles"]
                            + .13 * s["pigmentation"] + .12 * s["spots"] + .10 * (100 - s["symmetry"]))
    s["skin_health"] = 100 - s["aging_index"]
    s.update(forehead=fore, crows_feet=float(crow), nasolabial=float(nasol))

    # frame quality: sharpness x exposure x frontal pose
    x0, y0, x1, y1 = [int(v) for v in (*P.min(0), *P.max(0))]
    fc = gray[max(0, y0):y1, max(0, x0):x1]
    sharp = clip(np.log1p(cv2.Laplacian(fc, cv2.CV_64F).var()) / 6, 0, 1) if fc.size else 0
    br = float(L[max(0, y0):y1, max(0, x0):x1].mean() * 100 / 255) if fc.size else 50
    q = sharp * (1.0 if 35 < br < 80 else .5) * max(.05, 1 - abs(yaw) * 3)

    ages, males = deep_age_gender(img, P) if HAS_DL else ([], [])
    if not HAS_DL:
        males = [float(np.clip((abs(P[58, 0] - P[288, 0]) / fw - .76) * 8 + .5, .1, .9))]
        ages = [20 + s["aging_index"] * .55]
    raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[max(0, y0):y1, max(0, x0):x1]
    try: ffa = fairface_age(img, P) if FF else []
    except Exception as e: print("fairface skipped:", e, flush=True); ffa = []
    try: emo = expression(raw) if EM and raw.size else None
    except Exception as e: print("expression skipped:", e, flush=True); emo = None
    return s, ages, males, float(q), ffa, emo


def skin_heatmap_b64(img):
    """Create an illustrative per-region visible skin variation heatmap, not a diagnosis."""
    h, w = img.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    res = FL.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks: return None
    P = np.array([[l.x*w, l.y*h] for l in res.face_landmarks[0]], np.int32)
    mask = np.zeros((h,w), np.uint8); cv2.fillPoly(mask, [P[OVAL]], 255)
    for idx in (LIPS, LEYE, REYE): cv2.fillPoly(mask, [P[idx]], 0)
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycc, (0, 133, 77), (255, 173, 127))
    mask = cv2.erode(cv2.bitwise_and(mask, skin), np.ones((5,5),np.uint8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:,:,0].astype(np.float32); smooth=cv2.GaussianBlur(L,(0,0),7)
    chroma=cv2.absdiff(lab[:,:,1], np.full_like(lab[:,:,1],128)).astype(np.float32)
    raw=cv2.normalize(np.abs(L-smooth)*2.0+chroma*.35,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    raw=cv2.bitwise_and(raw,raw,mask=mask); heat=cv2.applyColorMap(raw,cv2.COLORMAP_TURBO)
    overlay=cv2.addWeighted(img,.48,heat,.52,0); overlay=cv2.bitwise_and(overlay,overlay,mask=cv2.dilate(mask,np.ones((9,9),np.uint8)))
    ok, buf=cv2.imencode('.jpg',overlay,[int(cv2.IMWRITE_JPEG_QUALITY),82])
    return base64.b64encode(buf).decode('ascii') if ok else None

@app.get("/")
def root():
    return jsonify(message="BIO-REVIVE 360 face-scan API online", dnn_models=HAS_DL, models=dict(genderage=HAS_DL, fairface=bool(FF), expression=bool(EM)))

@app.get("/scan")
def page():
    return send_from_directory(HERE, "face-scan.html")

@app.get("/models/<path:name>")
def local_model(name):
    # Restrict browser-accessible model files to the local models directory.
    allowed = {"face_landmarker.task", "genderage.onnx", "fairface.onnx", "emotion-ferplus-8.onnx",
               "pose_landmarker_lite.task", "pose_landmarker_full.task", "pose_landmarker_heavy.task"}
    if name not in allowed:
        return jsonify(error="Model not found"), 404
    return send_from_directory(MD, name, conditional=True)

@app.post("/api/face-scan")
def scan():
    d = request.get_json(force=True, silent=True) or {}
    srcs = (d.get("images") or [d.get("image")])[:12]; out = []
    for src in srcs:
        try:
            im = cv2.imdecode(np.frombuffer(base64.b64decode(src.split(",")[-1]), np.uint8), cv2.IMREAD_COLOR)
            r = analyze(im)
            if r: out.append(r)
        except Exception as e:
            print("frame skipped:", e, flush=True)
    if not out:
        return jsonify(detail="No face detected - come closer and face the light"), 422
    good = [r for r in out if r[3] >= .15] or out            # quality gate
    wts = np.array([r[3] + 1e-3 for r in good]); wts /= wts.sum()
    keys = good[0][0].keys()
    m = {k: float(sum(wt * r[0][k] for wt, r in zip(wts, good))) for k in keys}
    ages = np.array([a for r in good for a in r[1]]); males = np.array([x for r in good for x in r[2]])
    chrono = float(np.median(ages)); sd = float(ages.std()) if HAS_DL else 8.0
    pm = float(np.median(males))
    ffs = np.array([a for r in good for a in r[4]])
    if ffs.size:                                    # ensemble: genderage regression + FairFace expected age
        ff = float(np.median(ffs))
        if HAS_DL: sd = max(sd, abs(ff - chrono) / 2); chrono = .65 * chrono + .35 * ff
        else: chrono, sd = ff, 6.0
    half = max(4.0, 1.64 * sd) if HAS_DL else 10.0
    chrono_i = round(chrono)
    expected = float(np.clip((chrono - 18) * 1.15, 8, 85))
    bio = round(chrono + float(np.clip((m["aging_index"] - expected) * .2, -7, 12)))
    group = ("Child" if chrono < 13 else "Teen" if chrono < 20 else "Young adult" if chrono < 40 else "Adult" if chrono < 60 else "Senior")
    tips = []
    if m["dark_circles"] > 45: tips.append("Pronounced under-eye shadow: prioritise sleep and hydration.")
    if m["wrinkles"] > 50: tips.append("Visible fine lines: daily SPF and a retinoid (ask a dermatologist).")
    if m["pigmentation"] > 50 or m["spots"] > 50: tips.append("Uneven tone or spots: sun protection and vitamin C can help.")
    if m["redness"] > 50: tips.append("Elevated redness: use a gentle routine; see a doctor if persistent.")
    if m["shine"] > 50: tips.append("High shine: a lightweight, oil-free moisturiser may help.")
    emos = [r[5] for r in good if r[5]]; expr = None
    if emos:
        tally = {}
        for lab_, cf in emos: tally[lab_] = tally.get(lab_, 0) + cf
        expr = max(tally, key=tally.get)
        if expr != "neutral": tips.insert(0, f"Expression detected: {expr}. Rescan with a relaxed, neutral face for more accurate wrinkle readings.")
    return jsonify(chrono_age=chrono_i, expression=expr, models=dict(genderage=HAS_DL, fairface=bool(FF), expression=bool(EM)), age_range=[max(1, round(chrono - half)), round(chrono + half)],
                   age_conf=round(clip(95 - sd * 6, 40, 95)) if HAS_DL else 35,
                   age_group=group, frames_used=len(good), frames_sent=len(srcs),
                   engine="InsightFace genderage + FairFace age ensemble (ONNX, test-time augmentation) + MediaPipe 478 landmarks + OpenCV skin analysis" if HAS_DL
                   else "MediaPipe + OpenCV heuristic only (deep age model not loaded: low confidence)",
                   scores={k: round(v, 1) for k, v in m.items()}, tips=tips or ["Skin signals look balanced. Keep up sleep, SPF and hydration."],
                   heatmap=skin_heatmap_b64(cv2.imdecode(np.frombuffer(base64.b64decode(srcs[0].split(",")[-1]), np.uint8), cv2.IMREAD_COLOR)) if srcs and srcs[0] else None,
                   disclaimer="Appearance-based age estimate and visible-skin image metrics only; not biological age, organ age, gender identification, or a medical diagnosis. Model errors may be substantial.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
