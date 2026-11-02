#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VOGUE multi-camera crop-and-cut prototype.

Stage 1: detect the main subject and face in cam0/cam1/cam2.
Stage 2: create a push-in / camera-cut / push-in edit using synchronized frames.

The detector wrappers are loaded from:
  utils.cv_tools.Segmentation
  utils.face_detector_retina.FaceDetector
"""
from __future__ import annotations

import inspect
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("VOGUE")


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    """Convert common tensor/image containers to numpy without assuming a backend."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            value = value.numpy()
        except Exception:
            pass
    try:
        return np.asarray(value)
    except Exception:
        return None


def _as_float_box(box: Any, width: int, height: int) -> Optional[List[float]]:
    arr = _to_numpy(box)
    if arr is None:
        return None
    arr = np.asarray(arr).reshape(-1)
    if arr.size < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in arr[:4]]
    # Normalize common [0, 1] detector coordinates.
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    x1 = float(np.clip(x1, 0, width - 1))
    x2 = float(np.clip(x2, 0, width - 1))
    y1 = float(np.clip(y1, 0, height - 1))
    y2 = float(np.clip(y2, 0, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if mask.max() == 0:
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    if stats[largest, cv2.CC_STAT_AREA] < max(16, int(mask.size * 0.001)):
        return np.zeros_like(mask)
    return (labels == largest).astype(np.uint8)


def _extract_mask_candidate(value: Any, depth: int = 0) -> Optional[np.ndarray]:
    """Find a likely segmentation map inside dict/tuple/list model outputs."""
    if value is None or depth > 5:
        return None
    if isinstance(value, dict):
        priority = ("mask", "masks", "segmentation", "seg", "saliency",
                    "saliency_map", "pred", "prediction", "output", "out")
        for key in priority:
            if key in value:
                found = _extract_mask_candidate(value[key], depth + 1)
                if found is not None:
                    return found
        for item in value.values():
            found = _extract_mask_candidate(item, depth + 1)
            if found is not None:
                return found
        return None
    if isinstance(value, (tuple, list)):
        # Model wrappers often return (visualization, mask); inspect from the end.
        for item in reversed(value):
            found = _extract_mask_candidate(item, depth + 1)
            if found is not None:
                return found
        return None
    arr = _to_numpy(value)
    if arr is None or arr.ndim < 2:
        return None
    arr = np.asarray(arr)
    while arr.ndim > 2:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[0] in (2, 3, 4):
            # Channel-first class logits/probabilities.
            arr = arr[1] if arr.shape[0] == 2 else arr[:3].mean(axis=0)
        elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = cv2.cvtColor(arr[..., :3].astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            arr = np.squeeze(arr)
            if arr.ndim > 2:
                return None
    if arr.ndim != 2 or arr.size < 64:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    arr = np.nan_to_num(arr.astype(np.float32))
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return None
    if lo >= 0.0 and hi <= 1.0:
        mask = (arr >= 0.5).astype(np.uint8)
    elif lo >= 0.0 and hi <= 255.0:
        # Binary/byte masks are common outputs from segmentation wrappers.
        threshold = 127.0 if hi > 1.5 else 0.5
        mask = (arr >= threshold).astype(np.uint8)
    else:
        # Treat non-normalized outputs as logits.
        mask = (arr > 0.0).astype(np.uint8)
    return mask


def _extract_box_candidates(value: Any, width: int, height: int,
                            depth: int = 0) -> List[Tuple[List[float], float]]:
    """Extract xyxy boxes from common detector return formats."""
    if value is None or depth > 5:
        return []
    found: List[Tuple[List[float], float]] = []
    if isinstance(value, dict):
        for key in ("boxes", "bboxes", "face_boxes", "face_bboxes", "face_dets",
                    "dets", "detections", "bbox", "box", "face", "results"):
            if key in value:
                found.extend(_extract_box_candidates(value[key], width, height, depth + 1))
        if found:
            return found
        for item in value.values():
            found.extend(_extract_box_candidates(item, width, height, depth + 1))
        return found
    if isinstance(value, (tuple, list)):
        # A single box is often returned as a plain Python list [x1, y1, x2, y2, score].
        try:
            list_arr = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError):
            list_arr = None
        if list_arr is not None and list_arr.ndim == 1 and list_arr.size >= 4:
            box = _as_float_box(list_arr, width, height)
            if box:
                score = float(list_arr[4]) if list_arr.size >= 5 else 1.0
                return [(box, score)]
        for item in value:
            found.extend(_extract_box_candidates(item, width, height, depth + 1))
        return found
    arr = _to_numpy(value)
    if arr is None:
        return found
    arr = np.asarray(arr)
    if arr.ndim == 1 and arr.size >= 4:
        box = _as_float_box(arr, width, height)
        if box:
            score = float(arr[4]) if arr.size >= 5 else 1.0
            found.append((box, score))
        return found
    if arr.ndim == 2 and arr.shape[1] >= 4 and arr.shape[0] <= 100:
        for row in arr:
            box = _as_float_box(row, width, height)
            if box:
                score = float(row[4]) if row.size >= 5 else 1.0
                found.append((box, score))
    return found


def _call_model(model: Any, method_names: Sequence[str], frame_bgr: np.ndarray,
                label: str) -> Any:
    """Call the first compatible public inference method on a local wrapper."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = None
    try:
        from PIL import Image
        pil_image = Image.fromarray(rgb)
    except Exception:
        pass

    errors: List[str] = []
    attempted = set()
    for method_name in method_names:
        method = getattr(model, method_name, None)
        if not callable(method):
            continue
        for image in (frame_bgr, rgb, pil_image):
            if image is None:
                continue
            signature_key = (method_name, type(image).__name__)
            if signature_key in attempted:
                continue
            attempted.add(signature_key)
            try:
                return method(image)
            except TypeError as exc:
                errors.append(f"{method_name}({type(image).__name__}): {exc}")
            except Exception as exc:
                errors.append(f"{method_name}({type(image).__name__}): {type(exc).__name__}: {exc}")
                # A method may exist but expect another color/layout convention.
                continue

    available = [name for name in dir(model) if not name.startswith("_") and callable(getattr(model, name, None))]
    detail = "; ".join(errors[-4:]) if errors else "no known inference method was found"
    raise RuntimeError(
        f"Cannot run {label}. Tried methods {list(method_names)}. "
        f"Available callable methods: {available}. Last errors: {detail}. "
        "Please provide the exact inference method/signature from the local wrapper."
    )


def _interpolate_boxes(boxes: Sequence[Optional[Sequence[float]]]) -> List[Optional[List[float]]]:
    valid = [i for i, box in enumerate(boxes) if box is not None]
    if not valid:
        return [None for _ in boxes]
    values = np.asarray([boxes[i] for i in valid], dtype=np.float32)
    result = []
    xs = np.arange(len(boxes), dtype=np.float32)
    for i in range(len(boxes)):
        coords = [np.interp(xs[i], valid, values[:, k]) for k in range(4)]
        result.append([float(v) for v in coords])
    return result


def _smooth_boxes(boxes: Sequence[Optional[Sequence[float]]], radius: int = 2) -> List[Optional[List[float]]]:
    if not boxes:
        return []
    arr = np.asarray([b if b is not None else [np.nan] * 4 for b in boxes], dtype=np.float32)
    out = arr.copy()
    for i in range(len(arr)):
        lo, hi = max(0, i - radius), min(len(arr), i + radius + 1)
        window = arr[lo:hi]
        for k in range(4):
            valid = window[:, k][np.isfinite(window[:, k])]
            if valid.size:
                out[i, k] = float(np.median(valid))
    result = []
    for row in out:
        result.append([float(v) for v in row] if np.all(np.isfinite(row)) else None)
    return result


def _fit_crop_rect(frame_shape: Sequence[int], box: Sequence[float],
                   aspect: float, fill: float) -> Tuple[float, float, float, float]:
    """Return center-x, center-y, crop-width, crop-height that fits a bbox."""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(2.0, x2 - x1), max(2.0, y2 - y1)
    fill = float(np.clip(fill, 0.2, 0.98))
    crop_h = max(bh / fill, bw / (aspect * fill))
    crop_w = crop_h * aspect
    # Keep the requested crop fully inside the source whenever possible.
    scale = min(1.0, w / crop_w, h / crop_h)
    crop_w *= scale
    crop_h *= scale
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    cx = float(np.clip(cx, crop_w * 0.5, w - crop_w * 0.5))
    cy = float(np.clip(cy, crop_h * 0.5, h - crop_h * 0.5))
    return cx, cy, crop_w, crop_h


def _mix_rect(start: Tuple[float, float, float, float],
              end: Tuple[float, float, float, float],
              progress: float) -> Tuple[float, float, float, float]:
    p = float(np.clip(progress, 0.0, 1.0))
    # Smoothstep makes the push-in start/end less abrupt.
    p = p * p * (3.0 - 2.0 * p)
    cx = start[0] * (1.0 - p) + end[0] * p
    cy = start[1] * (1.0 - p) + end[1] * p
    cw = math.exp(math.log(max(1.0, start[2])) * (1.0 - p) +
                  math.log(max(1.0, end[2])) * p)
    ch = math.exp(math.log(max(1.0, start[3])) * (1.0 - p) +
                  math.log(max(1.0, end[3])) * p)
    return cx, cy, cw, ch


def _crop_resize(frame: np.ndarray, rect: Tuple[float, float, float, float],
                 output_size: Tuple[int, int]) -> np.ndarray:
    cx, cy, crop_w, crop_h = rect
    out_w, out_h = output_size
    x0 = cx - crop_w * 0.5
    y0 = cy - crop_h * 0.5
    # remap handles edge cases without changing the crop aspect ratio.
    map_x = (x0 + (np.arange(out_w, dtype=np.float32) + 0.5) * crop_w / out_w)
    map_y = (y0 + (np.arange(out_h, dtype=np.float32) + 0.5) * crop_h / out_h)
    grid_x, grid_y = np.meshgrid(map_x, map_y)
    return cv2.remap(frame, grid_x, grid_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


class _FrameReader:
    def __init__(self, path: Path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.last_index = -2

    def read(self, index: int) -> np.ndarray:
        index = max(0, int(index))
        if index != self.last_index + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"Failed reading frame {index} from {self.path}")
        self.last_index = index
        return frame

    def close(self) -> None:
        self.cap.release()


class VOGUEPipeline:
    CAMERA_ORDER = ("cam0", "cam1", "cam2")

    def __init__(
        self,
        case_dir: str,
        output_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        utils_root: Optional[str] = None,
        device: str = "cuda",
        detect_stride: int = 1,
        first_push_ratio: float = 1.0 / 3.0,
        cut_hold_seconds: float = 0.55,
        subject_fill: float = 0.86,
        upper_fill: float = 0.88,
        upper_body_ratio: float = 0.58,
    ):
        self.case_dir = Path(case_dir).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve() if output_dir else self.case_dir / "VOGUE_output"
        self.model_dir = Path(model_dir).expanduser().resolve() if model_dir else Path(__file__).resolve().parent / "Cloud_Models" / "Saliency-Face-DA3"
        self.utils_root = Path(utils_root).expanduser().resolve() if utils_root else Path(__file__).resolve().parent
        self.device_name = device
        self.detect_stride = max(1, int(detect_stride))
        self.first_push_ratio = float(np.clip(first_push_ratio, 0.1, 0.8))
        self.cut_hold_seconds = max(0.0, float(cut_hold_seconds))
        self.subject_fill = float(np.clip(subject_fill, 0.5, 0.97))
        self.upper_fill = float(np.clip(upper_fill, 0.5, 0.97))
        self.upper_body_ratio = float(np.clip(upper_body_ratio, 0.35, 0.8))
        self.video_paths = {name: self.case_dir / f"{name}.mp4" for name in self.CAMERA_ORDER}
        self.detections: Dict[str, Dict[str, Any]] = {}
        self.video_info: Dict[str, Dict[str, Any]] = {}

        if not self.case_dir.exists():
            raise FileNotFoundError(f"Case directory does not exist: {self.case_dir}")
        missing = [str(p) for p in self.video_paths.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError("Missing input videos:\n  " + "\n  ".join(missing))

        sys.path.insert(0, str(self.utils_root))
        try:
            import torch
            from utils.face_detector_retina import FaceDetector
            from utils.cv_tools import Segmentation
        except Exception as exc:
            raise ImportError(
                "Failed importing the detector wrappers. Expected "
                "utils.face_detector_retina.FaceDetector and utils.cv_tools.Segmentation. "
                f"utils_root={self.utils_root}. Original error: {exc}"
            ) from exc

        if device == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA is unavailable; falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        LOGGER.info("Initializing Segmentation from %s", self.model_dir)
        self.seg_module = Segmentation(str(self.model_dir), self.device)
        LOGGER.info("Initializing FaceDetector from %s", self.model_dir)
        self.face_detector = FaceDetector(0.8, 50, str(self.model_dir), self.device)

    def _video_metadata(self, path: Path) -> Dict[str, Any]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        info = {
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
        cap.release()
        if info["fps"] <= 0 or info["frame_count"] <= 0:
            raise RuntimeError(f"Invalid video metadata for {path}: {info}")
        info["duration"] = info["frame_count"] / info["fps"]
        return info

    def _segment_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
        output = _call_model(
            self.seg_module,
            ("segment", "get_mask", "get_segmentation", "get_saliency",
             "saliency", "inference", "predict", "process", "run", "forward", "__call__"),
            frame, "Segmentation"
        )
        h, w = frame.shape[:2]
        mask = _extract_mask_candidate(output)
        direct_boxes = _extract_box_candidates(output, w, h)
        if mask is not None:
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            mask = _largest_component(mask)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            ys, xs = np.where(mask > 0)
            if len(xs) >= max(20, int(h * w * 0.001)):
                box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
                return mask, box
        if direct_boxes:
            direct_boxes.sort(key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]), reverse=True)
            return None, direct_boxes[0][0]
        return None, None

    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[List[float], float]]:
        output = _call_model(
            self.face_detector,
            ("detect_image", "detect_faces", "detect_face", "get_faces",
             "detect", "inference", "predict", "process", "run", "forward", "__call__"),
            frame, "FaceDetector"
        )
        h, w = frame.shape[:2]
        candidates = _extract_box_candidates(output, w, h)
        candidates = [(box, score) for box, score in candidates if score >= 0.0]
        candidates.sort(key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]), reverse=True)
        return candidates

    def detect_all(self) -> None:
        """Run person segmentation and face detection on each source video."""
        detection_root = self.output_dir / "detection"
        mask_root = detection_root / "masks"
        mask_root.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for camera, path in self.video_paths.items():
            info = self._video_metadata(path)
            self.video_info[camera] = info
            LOGGER.info(
                "[%s] %d frames, %.3f FPS, %dx%d, %.2fs",
                camera, info["frame_count"], info["fps"], info["width"], info["height"], info["duration"]
            )
            cap = cv2.VideoCapture(str(path))
            subject_samples: List[Optional[List[float]]] = []
            face_samples: List[Optional[List[float]]] = []
            sample_indices: List[int] = []
            camera_mask_dir = mask_root / camera
            camera_mask_dir.mkdir(parents=True, exist_ok=True)
            frame_index = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % self.detect_stride == 0 or frame_index == info["frame_count"] - 1:
                    mask, subject_box = self._segment_frame(frame)
                    faces = self._detect_faces(frame)
                    face_box = faces[0][0] if faces else None
                    sample_indices.append(frame_index)
                    subject_samples.append(subject_box)
                    face_samples.append(face_box)
                    if mask is not None:
                        cv2.imwrite(str(camera_mask_dir / f"frame_{frame_index:06d}.png"), mask * 255)
                    if frame_index == 0 or frame_index == info["frame_count"] // 2:
                        preview = frame.copy()
                        if subject_box:
                            x1, y1, x2, y2 = [int(round(v)) for v in subject_box]
                            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 220, 0), 2)
                        if face_box:
                            x1, y1, x2, y2 = [int(round(v)) for v in face_box]
                            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.imwrite(str(detection_root / f"{camera}_preview_{frame_index:06d}.jpg"), preview)
                    if len(sample_indices) % 50 == 0:
                        LOGGER.info("[%s] detected %d sampled frames...", camera, len(sample_indices))
                frame_index += 1
            cap.release()

            if not sample_indices:
                raise RuntimeError(f"No frames were decoded from {path}")
            all_subject: List[Optional[List[float]]] = [None] * info["frame_count"]
            all_face: List[Optional[List[float]]] = [None] * info["frame_count"]
            for frame_idx, box in zip(sample_indices, _interpolate_boxes(subject_samples)):
                all_subject[frame_idx] = box
            for frame_idx, box in zip(sample_indices, _interpolate_boxes(face_samples)):
                all_face[frame_idx] = box

            # Interpolate across all source frames, then smooth to avoid crop jitter.
            sparse_subject = [None] * info["frame_count"]
            sparse_face = [None] * info["frame_count"]
            for frame_idx, box in zip(sample_indices, subject_samples):
                sparse_subject[frame_idx] = box
            for frame_idx, box in zip(sample_indices, face_samples):
                sparse_face[frame_idx] = box
            all_subject = _smooth_boxes(_interpolate_boxes(sparse_subject), radius=2)
            all_face = _smooth_boxes(_interpolate_boxes(sparse_face), radius=2)

            if not any(box is not None for box in all_subject):
                raise RuntimeError(
                    f"No subject was detected in {camera}. Check the Segmentation wrapper output/API."
                )
            self.detections[camera] = {
                "subject_bbox": all_subject,
                "face_bbox": all_face,
                "sampled_frame_indices": sample_indices,
                "mask_dir": str(camera_mask_dir),
            }
            payload = {
                "camera": camera,
                "video": str(path),
                "video_info": info,
                "detect_stride": self.detect_stride,
                "subject_bbox_xyxy_per_frame": all_subject,
                "face_bbox_xyxy_per_frame": all_face,
                "sampled_frame_indices": sample_indices,
                "mask_dir": str(camera_mask_dir),
            }
            with (detection_root / f"{camera}_detections.json").open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            LOGGER.info("[%s] detection complete; masks and JSON saved.", camera)

        with (detection_root / "all_detections.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"video_info": self.video_info, "detections": self.detections},
                f, ensure_ascii=False, indent=2
            )

    @staticmethod
    def _bbox_at(camera_data: Dict[str, Any], key: str, frame_idx: int) -> Optional[List[float]]:
        boxes = camera_data[key]
        if not boxes:
            return None
        idx = int(np.clip(frame_idx, 0, len(boxes) - 1))
        return boxes[idx]

    def _upper_body_box(self, person_box: Sequence[float],
                        face_box: Optional[Sequence[float]]) -> List[float]:
        x1, y1, x2, y2 = [float(v) for v in person_box]
        height = max(2.0, y2 - y1)
        upper_y2 = y1 + self.upper_body_ratio * height
        if face_box is not None:
            upper_y2 = max(upper_y2, float(face_box[3]) + 0.12 * height)
        upper_y2 = min(y2, upper_y2)
        # Preserve the shoulder width while trimming the lower body.
        return [x1, y1, x2, upper_y2]

    def render(self, output_name: str = "VOGUE_output.mp4") -> Path:
        """Render back full-frame -> subject framing -> three-view cut -> upper-body push-in."""
        if not self.detections or not self.video_info:
            raise RuntimeError("Call detect_all() before render().")

        duration = min(self.video_info[c]["duration"] for c in self.CAMERA_ORDER)
        output_fps = self.video_info["cam2"]["fps"]
        output_w = self.video_info["cam2"]["width"]
        output_h = self.video_info["cam2"]["height"]
        aspect = output_w / float(output_h)
        hold_time = duration * self.first_push_ratio
        hold_frame = max(1, int(round(hold_time * output_fps)))
        total_frames = max(hold_frame + 1, int(round(duration * output_fps)))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "image"
        image_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / output_name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (output_w, output_h))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {output_path}")

        readers = {c: _FrameReader(self.video_paths[c]) for c in self.CAMERA_ORDER}
        try:
            LOGGER.info("Rendering first push-in from cam2 (back view)...")
            for out_idx in range(hold_frame):
                t = out_idx / output_fps
                src_idx = int(round(t * self.video_info["cam2"]["fps"]))
                frame = readers["cam2"].read(src_idx)
                person = self._bbox_at(self.detections["cam2"], "subject_bbox", src_idx)
                if person is None:
                    raise RuntimeError(f"cam2 subject bbox missing at frame {src_idx}")
                target = _fit_crop_rect(frame.shape, person, aspect, self.subject_fill)
                full = (output_w * 0.5, output_h * 0.5, float(output_w), float(output_h))
                progress = min(1.0, t / max(1e-6, hold_time))
                rect = _mix_rect(full, target, progress)
                writer.write(_crop_resize(frame, rect, (output_w, output_h)))

            # Freeze the synchronized moment and save the three aligned camera frames.
            LOGGER.info("Saving synchronized camera-cut frames at t=%.3fs...", hold_time)
            cut_frames: Dict[str, np.ndarray] = {}
            for camera, label in (("cam2", "back"), ("cam1", "right"), ("cam0", "front")):
                src_idx = int(round(hold_time * self.video_info[camera]["fps"]))
                frame = readers[camera].read(src_idx)
                person = self._bbox_at(self.detections[camera], "subject_bbox", src_idx)
                if person is None:
                    raise RuntimeError(f"{camera} subject bbox missing at synchronized frame {src_idx}")
                rect = _fit_crop_rect(frame.shape, person, aspect, self.subject_fill)
                cropped = _crop_resize(frame, rect, (output_w, output_h))
                cut_frames[camera] = cropped
                cv2.imwrite(str(image_dir / f"{camera}_{label}_subject_framing.png"), cropped)
                # Save source frames as well, so framing can be revised without re-decoding.
                cv2.imwrite(str(image_dir / f"{camera}_{label}_original.png"), frame)

            with (image_dir / "cut_frame_metadata.json").open("w", encoding="utf-8") as f:
                json.dump({
                    "sync_time_seconds": hold_time,
                    "source_frame_indices": {
                        c: int(round(hold_time * self.video_info[c]["fps"])) for c in self.CAMERA_ORDER
                    },
                    "camera_order": ["cam2", "cam1", "cam0"],
                    "files": {
                        c: f"{c}_{label}_subject_framing.png"
                        for c, label in (("cam2", "back"), ("cam1", "right"), ("cam0", "front"))
                    },
                }, f, ensure_ascii=False, indent=2)

            cut_repeat = max(1, int(round(self.cut_hold_seconds * output_fps)))
            for camera in ("cam2", "cam1", "cam0"):
                for _ in range(cut_repeat):
                    writer.write(cut_frames[camera])

            LOGGER.info("Rendering second push-in from cam0 (front view)...")
            remaining = max(1, total_frames - hold_frame)
            for out_idx in range(hold_frame, total_frames):
                t = out_idx / output_fps
                src_idx = int(round(t * self.video_info["cam0"]["fps"]))
                frame = readers["cam0"].read(src_idx)
                person = self._bbox_at(self.detections["cam0"], "subject_bbox", src_idx)
                face = self._bbox_at(self.detections["cam0"], "face_bbox", src_idx)
                if person is None:
                    raise RuntimeError(f"cam0 subject bbox missing at frame {src_idx}")
                upper = self._upper_body_box(person, face)
                start_rect = _fit_crop_rect(frame.shape, person, aspect, self.subject_fill)
                end_rect = _fit_crop_rect(frame.shape, upper, aspect, self.upper_fill)
                progress = (out_idx - hold_frame) / max(1, remaining - 1)
                rect = _mix_rect(start_rect, end_rect, progress)
                writer.write(_crop_resize(frame, rect, (output_w, output_h)))
        finally:
            writer.release()
            for reader in readers.values():
                reader.close()

        LOGGER.info("Rendered video: %s", output_path)
        LOGGER.info("Synchronized frames: %s", image_dir)
        return output_path

    def run(self) -> Path:
        self.detect_all()
        return self.render()
