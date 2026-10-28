"""Reliable BIO-REVIVE model downloader.

Default: face landmarks + InsightFace age/gender ONNX + MediaPipe full-body pose.
Use --all to add optional age ensemble/expression and Lite/Heavy pose variants.
Use --legacy-caffe to fetch the original Levi-Hassner Caffe age/gender files.
Files are downloaded to ./models with retries, partial-file cleanup, size checks,
and SHA-256 verification when a digest is configured.
"""
import argparse
import hashlib
import os
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "models")

FACE = {
    "face_landmarker.task": {
        "urls": ["https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"],
        "min_size": 1_000_000,
        "sha256": None,
        "purpose": "478-point face landmarks",
    },
    "genderage.onnx": {
        "urls": [
            "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/genderage.onnx",
            "https://huggingface.co/luca115/stable-diffusion/resolve/main/insightface/models/buffalo_l/genderage.onnx",
        ],
        "min_size": 100_000,
        "sha256": "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb",
        "purpose": "experimental apparent-age/gender presentation model; check model licence before distribution",
    },
    "pose_landmarker_full.task": {
        "urls": ["https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"],
        "min_size": 3_000_000,
        "sha256": None,
        "purpose": "33-landmark 3D body pose; recommended default",
    },
}
OPTIONAL = {
    "fairface.onnx": {
        "urls": [
            "https://huggingface.co/bluefoxcreation/gender_age/resolve/main/fair_face.onnx",
            "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/fairface.onnx",
        ], "min_size": 80_000_000, "sha256": None,
        "purpose": "optional age-bucket ensemble; source/redistribution licence must be reviewed",
    },
    "emotion-ferplus-8.onnx": {
        "urls": [
            "https://github.com/spmallick/learnopencv/raw/master/Facial-Emotion-Recognition/emotion-ferplus-8.onnx",
            "https://huggingface.co/webai-community/models/resolve/dd196431d06ee2f71ee45ea76d876e75949992bb/emotion-ferplus-8.onnx",
        ], "min_size": 30_000_000, "sha256": "a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c",
        "purpose": "optional facial-expression model; not a health measure",
    },
    "pose_landmarker_lite.task": {
        "urls": ["https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"],
        "min_size": 1_000_000, "sha256": None, "purpose": "faster/lighter 33-landmark pose option",
    },
    "pose_landmarker_heavy.task": {
        "urls": ["https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"],
        "min_size": 5_000_000, "sha256": None, "purpose": "higher-capacity/slower 33-landmark pose option",
    },
}
LEGACY_CAFFE = {
    "age_deploy.prototxt": {"urls": ["https://raw.githubusercontent.com/smahesh29/gender-and-age-detection/master/age_deploy.prototxt"], "min_size": 1000, "sha256": None, "purpose": "legacy Caffe age model config (8 age bands)"},
    "age_net.caffemodel": {"urls": ["https://raw.githubusercontent.com/smahesh29/gender-and-age-detection/master/age_net.caffemodel"], "min_size": 1_000_000, "sha256": None, "purpose": "legacy Caffe age model weights"},
    "gender_deploy.prototxt": {"urls": ["https://raw.githubusercontent.com/smahesh29/gender-and-age-detection/master/gender_deploy.prototxt"], "min_size": 1000, "sha256": None, "purpose": "legacy Caffe binary gender-presentation config"},
    "gender_net.caffemodel": {"urls": ["https://raw.githubusercontent.com/smahesh29/gender-and-age-detection/master/gender_net.caffemodel"], "min_size": 1_000_000, "sha256": None, "purpose": "legacy Caffe binary gender-presentation weights"},
}

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def valid(path, spec):
    if not os.path.isfile(path) or os.path.getsize(path) < spec["min_size"]:
        return False
    expected = spec.get("sha256")
    return not expected or sha256(path).lower() == expected.lower()

def fetch(name, spec, retries=2):
    path = os.path.join(DEST, name)
    if valid(path, spec):
        print(f"[OK] {name} already present ({os.path.getsize(path):,} bytes)")
        return True
    if os.path.exists(path):
        print(f"[WARN] Existing {name} failed size/checksum validation; re-downloading")
        try: os.remove(path)
        except OSError: pass
    for url in spec["urls"]:
        for attempt in range(1, retries + 2):
            part = path + ".part"
            try:
                print(f"[GET] {name} ({spec['purpose']}) — attempt {attempt}", flush=True)
                req = urllib.request.Request(url, headers={"User-Agent": "BIO-REVIVE-model-setup/2.0"})
                with urllib.request.urlopen(req, timeout=90) as response, open(part, "wb") as dst:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block: break
                        dst.write(block)
                if not valid(part, spec):
                    raise ValueError(f"download too small or SHA-256 mismatch (size={os.path.getsize(part):,})")
                os.replace(part, path)
                print(f"[OK] {name} downloaded ({os.path.getsize(path):,} bytes)")
                return True
            except Exception as exc:
                print(f"[WARN] {name}: {exc}", flush=True)
                try:
                    if os.path.exists(part): os.remove(part)
                except OSError: pass
                if attempt <= retries: time.sleep(attempt * 1.5)
    print(f"[FAILED] {name}. Check network access, URL availability, model licence and try again.")
    return False

def ensure(profile="default", legacy_caffe=False):
    os.makedirs(DEST, exist_ok=True)
    selected = dict(FACE)
    if profile == "all": selected.update(OPTIONAL)
    if legacy_caffe: selected.update(LEGACY_CAFFE)
    results = {name: fetch(name, spec) for name, spec in selected.items()}
    print("\nModel setup summary:")
    for name, ok in results.items(): print(f"  {'READY' if ok else 'MISSING'}  {name}")
    return all(results.values())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="download optional FairFace/FER+ and Lite/Heavy pose variants too")
    parser.add_argument("--legacy-caffe", action="store_true", help="also fetch original Levi-Hassner Caffe age/gender files (not used by current ONNX inference code)")
    args = parser.parse_args()
    ok = ensure("all" if args.all else "default", args.legacy_caffe)
    raise SystemExit(0 if ok else 1)
