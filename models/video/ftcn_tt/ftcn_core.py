"""FTCN+TT inference (ported from FTCN repo test_on_raw_video.py).

Runs the full FTCN+TT pipeline: RetinaFace face detection -> 68 landmarks ->
multi-face tracking -> 32-frame clip sampling -> I3D temporal-transformer
classifier. The returned score is the mean classifier output over all clips,
higher value = more likely FAKE (matches the author's test loop).
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "8")

import logging
import threading
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch

from config import config as cfg
from utils.plugin_loader import PluginLoader

from test_tools.ct.detection import FaceDetector, get_valid_faces
from test_tools.ct.detection.utils import grab_all_frames
from test_tools.ct.face_alignment import LandmarkPredictor
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.utils import flatten, get_crop_box, partition

logger = logging.getLogger(__name__)

MAX_FRAME = int(os.environ.get("FTCN_MAX_FRAME", "240"))
NUM_THREADS = int(os.environ.get("FTCN_NUM_THREADS", "8"))

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_ID = 0 if _DEVICE.type == "cuda" else -1

mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(1, 3, 1, 1, 1)
std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(1, 3, 1, 1, 1)

_LOAD_LOCK = threading.Lock()
_loaded = False
_cls = None
_crop_align = None
_detector = None
_predictor = None


def _get_five(ldm68: np.ndarray) -> np.ndarray:
    groups = [range(36, 42), range(42, 48), [30], [48], [54]]
    points = []
    for group in groups:
        points.append(ldm68[group].mean(0))
    return np.array(points)


def _get_lm68(frames: List[np.ndarray], detect_res: List[List[Any]]) -> List[List[Any]]:
    all_68 = []
    for i in range(len(frames)):
        frame = frames[i]
        faces = detect_res[i]
        if len(faces) == 0:
            all_68.append([])
            continue
        feeds = []
        for face in faces:
            feeds.append(LandmarkPredictor.prepare_feed(frame, face[0]))
        res_68 = _predictor(feeds)
        assert len(res_68) == len(faces)
        for face, l_68 in zip(faces, res_68):
            if face[1] is None:
                face[1] = _get_five(l_68)
        all_68.append(res_68)
    return all_68


def _ensure_loaded() -> None:
    global _loaded, _cls, _crop_align, _detector, _predictor
    if _loaded:
        return
    with _LOAD_LOCK:
        if _loaded:
            return
        logger.info("Initializing FTCN+TT (%s)...", _DEVICE.type.upper())
        cfg.init_with_yaml()
        cfg.update_with_yaml("ftcn_tt.yaml")
        cfg.freeze()

        torch.set_num_threads(NUM_THREADS)
        classifier = PluginLoader.get_classifier(cfg.classifier_type)()
        classifier.eval()
        classifier.load("checkpoints/ftcn_tt.pth")
        if _DEVICE.type == "cuda":
            classifier = classifier.to(_DEVICE)
            logger.info("FTCN+TT classifier moved to CUDA")

        _cls = classifier
        _crop_align = FasterCropAlignXRay(cfg.imsize)
        _detector = FaceDetector(GPU_ID)
        _predictor = LandmarkPredictor(GPU_ID)
        _loaded = True
        logger.info("FTCN+TT initialized (max_frame=%d, threads=%d)", MAX_FRAME, NUM_THREADS)


def run_inference(video_path: str) -> Dict[str, Any]:
    """Run FTCN+TT on a raw video file and return the classifier score."""
    _ensure_loaded()

    frames = grab_all_frames(video_path, max_size=MAX_FRAME, cvt=True)
    if not frames:
        raise ValueError(f"No frames could be read from {video_path}")

    logger.info("Detecting faces on %d frames...", len(frames))
    detect_res = flatten([_detector.detect(item) for item in partition(frames, 50)])
    detect_res = get_valid_faces(detect_res, thres=0.5)
    all_68 = _get_lm68(frames, detect_res)
    assert len(all_68) == len(detect_res) == len(frames)

    frames_with_faces = sum(1 for f in detect_res if len(f) > 0)
    if frames_with_faces == 0:
        raise ValueError("No faces detected in video")

    shape = frames[0].shape[:2]

    all_detect_res = []
    for faces, faces_lm68 in zip(detect_res, all_68):
        new_faces = []
        for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
            new_faces.append((box, lm5, face_lm68, score))
        all_detect_res.append(new_faces)
    detect_res = all_detect_res

    logger.info("Splitting into super clips...")
    tracks = multiple_tracking(detect_res)
    tuples = [(0, len(detect_res))] * len(tracks)
    if len(tracks) == 0:
        raise ValueError("No face tracks found in video")

    data_storage: Dict[str, Any] = {}
    frame_boxes: Dict[int, np.ndarray] = {}
    super_clips = []

    for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
        assert len(detect_res[start:end]) == len(track)
        super_clips.append(len(track))

        for face, frame_idx, j in zip(track, range(start, end), range(len(track))):
            box, lm5, lm68 = face[:3]
            big_box = get_crop_box(shape, box, scale=0.5)
            top_left = big_box[:2][None, :]
            new_lm5 = lm5 - top_left
            new_lm68 = lm68 - top_left
            new_box = (box.reshape(2, 2) - top_left).reshape(-1)
            info = (new_box, new_lm5, new_lm68, big_box)

            x1, y1, x2, y2 = big_box
            cropped = frames[frame_idx][y1:y2, x1:x2]

            base_key = f"{track_i}_{j}_"
            data_storage[base_key + "img"] = cropped
            data_storage[base_key + "ldm"] = info
            data_storage[base_key + "idx"] = frame_idx

            frame_boxes[frame_idx] = np.rint(box).astype(np.int)

    logger.info("Sampling clips from super clips: %s", super_clips)
    clip_size = cfg.clip_size
    pad_length = clip_size - 1

    clips_for_video = []
    for super_clip_idx, super_clip_size in enumerate(super_clips):
        inner_index = list(range(super_clip_size))
        if super_clip_size < clip_size:
            post_module = inner_index[1:-1][::-1] + inner_index
            l_post = len(post_module)
            post_module = post_module * (pad_length // l_post + 1)
            post_module = post_module[:pad_length]
            assert len(post_module) == pad_length

            pre_module = inner_index + inner_index[1:-1][::-1]
            l_pre = len(post_module)
            pre_module = pre_module * (pad_length // l_pre + 1)
            pre_module = pre_module[-pad_length:]
            assert len(pre_module) == pad_length

            inner_index = pre_module + inner_index + post_module

        super_clip_size = len(inner_index)
        stride = 3
        frame_range = [
            inner_index[i : i + clip_size]
            for i in range(0, super_clip_size - clip_size + 1, stride)
        ]
        if not frame_range and super_clip_size >= clip_size:
            frame_range = [inner_index[:clip_size]]
        for indices in frame_range:
            clips_for_video.append([(super_clip_idx, t) for t in indices])

    # Cap maximum total clips evaluated per video to avoid extreme processing times & timeouts
    MAX_CLIPS_CAP = 150
    if len(clips_for_video) > MAX_CLIPS_CAP:
        step = len(clips_for_video) / MAX_CLIPS_CAP
        sampled_indices = [int(i * step) for i in range(MAX_CLIPS_CAP)]
        clips_for_video = [clips_for_video[idx] for idx in sampled_indices]

    preds = []
    frame_res: Dict[int, List[float]] = {}

    logger.info("Evaluating %d clips...", len(clips_for_video))
    for clip in clips_for_video:
        images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
        landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
        frame_ids = [data_storage[f"{i}_{j}_idx"] for i, j in clip]

        landmarks, images = _crop_align(landmarks, images)
        images_t = torch.as_tensor(images, dtype=torch.float32).permute(3, 0, 1, 2)
        images_t = images_t.unsqueeze(0).to(_DEVICE).sub(mean.to(_DEVICE)).div(std.to(_DEVICE))

        with torch.no_grad():
            output = _cls(images_t)
        pred = float(output["final_output"])
        for f_id in frame_ids:
            if f_id not in frame_res:
                frame_res[f_id] = []
            frame_res[f_id].append(pred)
        preds.append(pred)

    score = float(np.mean(preds))
    detail = {
        "total_frames_in_video": len(frames),
        "frames_analyzed": len(frames),
        "frames_with_faces": frames_with_faces,
        "super_clips": len(super_clips),
        "super_clip_sizes": super_clips,
        "clips_evaluated": len(clips_for_video),
        "max_frame_cap": MAX_FRAME,
    }
    logger.info("FTCN+TT score %.4f over %d clips", score, len(clips_for_video))
    return {"probability": score, "details": detail}