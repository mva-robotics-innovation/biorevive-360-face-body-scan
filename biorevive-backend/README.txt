BIO-REVIVE 360 — Face + Whole-Body Computer-Vision Lab

WINDOWS QUICK START
1. Extract this ZIP.
2. Open the biorevive-backend folder.
3. Run run.bat (first setup downloads model files; allow an internet connection).
4. Open http://localhost:5000/scan and allow camera access.

MODEL DOWNLOADER
- python download_models.py              downloads face landmarks, InsightFace genderage ONNX, and MediaPipe Pose Landmarker FULL.
- python download_models.py --all        also downloads FairFace age ensemble, FER+ expression, and Pose Lite + Heavy variants.
- python download_models.py --legacy-caffe also downloads the original Levi-Hassner Caffe age/gender files. Those legacy files are not used by the current ONNX inference path.
- Downloads are resumable by re-running the script, use temporary .part files, size checks, retries, and SHA-256 checks where a digest is known.

FULL-BODY MODEL OPTIONS (official MediaPipe model family)
- pose_landmarker_lite.task: lighter/faster, 33 pose landmarks.
- pose_landmarker_full.task: balanced default, 33 pose landmarks with approximate 3D coordinates.
- pose_landmarker_heavy.task: larger/slower alternative, 33 pose landmarks.
The page uses the local FULL model at /models/pose_landmarker_full.task after setup. To switch variants, change that path in face-scan.html and download the chosen model first.

RESEARCH NOTES / LIMITATIONS
MediaPipe Pose estimates visible body landmarks and posture. It does not see internal organs, determine organ age, diagnose disease, or calculate biological age. The face models estimate visible appearance/age bands and have demographic and lighting-related errors; gender outputs from face classifiers are inferred presentation labels, not gender identity. Do not use for medical, employment, eligibility, or other high-impact decisions. Review each model's licence and terms before redistribution or commercial deployment.

Sources for model families:
- MediaPipe Pose Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
- MediaPipe model assets: https://storage.googleapis.com/mediapipe-models/
- MoveNet alternative (17 keypoints): https://www.tensorflow.org/hub/tutorials/movenet
- OpenMMLab RTMPose / whole-body option: https://github.com/open-mmlab/mmpose

CHECK API
http://localhost:5000/ should return JSON. If dnn_models is false, inspect the model downloader output and the models/ folder.
