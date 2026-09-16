"""LipForensics video inference on CPU.

Implements the official LipForensics pipeline for a raw video:
  frames -> (S3FD face detection + FAN 68 landmarks via face_alignment)
  -> 12-frame smoothed alignment to LRW mean face (256x256)
  -> 96x96 mouth crop centred on landmarks[48:68]
  -> grayscale, /255, CenterCrop(88,88), normalize (0.421, 0.165)
  -> Lipreading (3D frontend + ResNet18 + MS-TCN) over non-overlapping
    25-frame clips -> mean logit -> sigmoid = fake probability.
"""
import os
import json
import threading
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import cv2
import torch
import face_alignment
from torchvision.transforms import Compose, CenterCrop

from models.spatiotemporal_net import Lipreading
from preprocessing.utils import warp_img, apply_transform, cut_patch
from data.transforms import ToTensorVideo, NormalizeVideo

logger = logging.getLogger(__name__)

os.environ.setdefault("OMP_NUM_THREADS", "8")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LIB_DIR = os.environ.get("LIP_MODEL_DIR", "/app/model")
CKPT_PATH = os.environ.get("LIP_CHECKPOINT", "/app/weights/lipforensics_ff.pth")
MEAN_FACE_PATH = os.path.join(LIB_DIR, "preprocessing", "20words_mean_face.npy")
CONFIG_PATH = os.path.join(LIB_DIR, "models", "configs", "lrw_resnet18_mstcn.json")

MAX_FRAMES = int(os.environ.get("LIP_MAX_FRAMES", "110"))
FRAMES_PER_CLIP = int(os.environ.get("LIP_CLIP_SIZE", "25"))
NUM_THREADS = int(os.environ.get("LIP_NUM_THREADS", "8"))
DETECT_MAX_SIDE = int(os.environ.get("LIP_DETECT_MAX_SIDE", "640"))
DETECT_CHUNK = int(os.environ.get("LIP_DETECT_CHUNK", "32"))

STD_SIZE = (256, 256)
STABLE_POINTS = [33, 36, 39, 42, 45]
WINDOW_MARGIN = 12
START_IDX, STOP_IDX = 48, 68
CROP_HALF = 48  # cut_patch gets half-size -> 96x96 mouth crop

_LOAD_LOCK = threading.Lock()
_loaded = False
_model: Optional[Lipreading] = None
_fa: Optional[face_alignment.FaceAlignment] = None
_mean_face: Optional[np.ndarray] = None
_tf: Optional[Compose] = None


def _ensure_loaded() -> None:
    global _loaded, _model, _fa, _mean_face, _tf
    if _loaded:
        return
    with _LOAD_LOCK:
        if _loaded:
            return
        logger.info("Initializing LipForensics (CPU)...")
        torch.set_num_threads(NUM_THREADS)

        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        model = Lipreading(
            num_classes=1,
            relu_type=cfg["relu_type"],
            tcn_options={
                "num_layers": cfg["tcn_num_layers"],
                "kernel_size": cfg["tcn_kernel_size"],
                "dropout": cfg["tcn_dropout"],
                "dwpw": cfg["tcn_dwpw"],
                "width_mult": cfg["tcn_width_mult"],
            },
        )
        state = torch.load(CKPT_PATH, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if any(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()
        if DEVICE == "cuda":
            model = model.to(DEVICE)
            logger.info("Lipreading model moved to CUDA")

        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=DEVICE,
            face_detector="sfd",
            flip_input=False,
        )

        _model, _fa, _mean_face, _tf = (
            model,
            fa,
            np.load(MEAN_FACE_PATH),
            Compose(
                [
                    ToTensorVideo(),
                    CenterCrop((88, 88)),
                    NormalizeVideo((0.421,), (0.165,)),
                ]
            ),
        )
        _loaded = True
        logger.info(
            "LipForensics initialized (max_frames=%d, clip_size=%d, threads=%d)",
            MAX_FRAMES,
            FRAMES_PER_CLIP,
            NUM_THREADS,
        )


def _read_frames(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []
    while cap.isOpened() and len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _smooth_landmarks(lm_all: List[Optional[np.ndarray]], i: int) -> Optional[np.ndarray]:
    win = [lm for lm in lm_all[max(0, i - WINDOW_MARGIN + 1): i + 1] if lm is not None]
    if not win:
        return None
    return np.mean(win, axis=0)


def _crop_mouths(rgb_frames: List[np.ndarray]) -> List[Optional[np.ndarray]]:
    h, w = rgb_frames[0].shape[:2]
    scale = min(DETECT_MAX_SIDE / float(max(h, w)), 1.0)
    frames = rgb_frames
    if scale < 1.0:
        tw, th = int(round(w * scale)), int(round(h * scale))
        frames = [
            cv2.resize(f, (tw, th), interpolation=cv2.INTER_LINEAR) for f in rgb_frames
        ]

    lms_list: List[Optional[List[np.ndarray]]] = []
    for i in range(0, len(frames), DETECT_CHUNK):
        chunk = frames[i:i + DETECT_CHUNK]
        batch = torch.from_numpy(np.stack(chunk)).permute(0, 3, 1, 2)
        lms_list.extend(_fa.get_landmarks_from_batch(batch) or [])

    lm_all: List[Optional[np.ndarray]] = []
    for x in lms_list:
        lm = None
        if x is not None and len(x) > 0:
            arr = np.asarray(x).reshape(-1, 68, 2)
            lm = np.float32(arr[0])
        if lm is not None and scale < 1.0:
            lm = lm / scale
        lm_all.append(lm)

    mouths: List[Optional[np.ndarray]] = []
    for i in range(len(rgb_frames)):
        sm = _smooth_landmarks(lm_all, i)
        if sm is None:
            mouths.append(None)
            continue
        warped, tform = warp_img(
            sm[STABLE_POINTS], _mean_face[STABLE_POINTS], rgb_frames[i], STD_SIZE
        )
        trans_lm = tform(sm)
        crop = cut_patch(warped, trans_lm[START_IDX:STOP_IDX], CROP_HALF, CROP_HALF)
        mouths.append(crop.astype(np.uint8))
    return mouths


def _fill_missing(mouths: List[Optional[np.ndarray]]) -> List[np.ndarray]:
    filled: List[np.ndarray] = []
    last_good: Optional[np.ndarray] = None
    for m in mouths:
        if m is None:
            if last_good is None:
                raise ValueError("No mouth crops available at the start of the video")
            m = last_good
        last_good = m
        filled.append(m)
    return filled


def run_inference(video_path: str) -> Dict[str, Any]:
    """Run LipForensics on a raw video and return the fake probability."""
    _ensure_loaded()

    rgb_frames = _read_frames(video_path)
    if not rgb_frames:
        raise ValueError(f"No frames could be read from {video_path}")

    logger.info("Detecting faces + landmarks on %d frames...", len(rgb_frames))
    mouths = _crop_mouths(rgb_frames)
    frames_with_mouth = sum(1 for m in mouths if m is not None)
    if frames_with_mouth == 0:
        raise ValueError("No faces detected in video")

    logger.info("Cropped mouths on %d/%d frames, filling gaps...", frames_with_mouth, len(rgb_frames))
    mouths = _fill_missing(mouths)

    gray = [np.mean(m.astype(np.float32), axis=2, keepdims=True).astype(np.uint8) for m in mouths]
    x = torch.from_numpy(np.asarray(gray))  # (T,H,W,1)
    x = _tf(x)  # (1,T,88,88)

    T = x.shape[1]
    clips = []
    if T >= FRAMES_PER_CLIP:
        clips = [
            x[:, i : i + FRAMES_PER_CLIP]
            for i in range(0, T - FRAMES_PER_CLIP + 1, FRAMES_PER_CLIP)
        ]
        if not clips:
            clips = [x[:, :FRAMES_PER_CLIP]]
    else:
        pad = x[:, :1].repeat(1, FRAMES_PER_CLIP - T, 1, 1)
        clips = [torch.cat([x, pad], dim=1)]

    logits: List[float] = []
    with torch.no_grad():
        for c in clips:
            out = _model(c.unsqueeze(0).to(DEVICE), lengths=[c.shape[1]])
            logits.append(float(out.squeeze(0).squeeze(0).item()))

    prob = float(torch.sigmoid(torch.tensor(logits).mean()).item())
    detail = {
        "total_frames_in_video": len(rgb_frames),
        "frames_analyzed": T,
        "frames_with_faces": frames_with_mouth,
        "clips_evaluated": len(clips),
        "clip_size": FRAMES_PER_CLIP,
        "max_frame_cap": MAX_FRAMES,
    }
    logger.info("LipForensics score %.4f over %d clips", prob, len(clips))
    return {"probability": prob, "details": detail}