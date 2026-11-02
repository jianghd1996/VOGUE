#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entry point for the first VOGUE multi-camera prototype."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pipeline_VOGUE import VOGUEPipeline


DEFAULT_CASE_DIR = "/home/z00566689/dev/mnt/jiang_test/00_1102_VOGUE/Case_13"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect the subject/face in three synchronized cameras and render a VOGUE-style push-in edit."
    )
    parser.add_argument("--case-dir", type=str, default=DEFAULT_CASE_DIR,
                        help="Folder containing cam0.mp4, cam1.mp4 and cam2.mp4.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output folder. Default: <case-dir>/VOGUE_output.")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Saliency-Face-DA3 model directory. Default: <script-dir>/Cloud_Models/Saliency-Face-DA3.")
    parser.add_argument("--utils-root", type=str, default=None,
                        help="Directory that contains the utils/ package.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device, e.g. cuda, cuda:0 or cpu.")
    parser.add_argument("--detect-stride", type=int, default=2,
                        help="Run segmentation/face detection every N frames; boxes are interpolated between samples.")
    parser.add_argument("--first-push-ratio", type=float, default=1.0 / 3.0,
                        help="Fraction of the source duration used for the first cam2 push-in.")
    parser.add_argument("--cut-hold-seconds", type=float, default=0.55,
                        help="Hold duration for each of the cam2 -> cam1 -> cam0 synchronized cut frames.")
    parser.add_argument("--subject-fill", type=float, default=0.86,
                        help="How much of the crop's limiting dimension the full subject occupies.")
    parser.add_argument("--upper-fill", type=float, default=0.88,
                        help="How much of the crop's limiting dimension the upper body occupies at the end.")
    parser.add_argument("--upper-body-ratio", type=float, default=0.58,
                        help="Upper-body bbox height as a fraction of the detected full-body bbox.")
    parser.add_argument("--output-name", type=str, default="VOGUE_output.mp4",
                        help="Rendered video filename.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    pipeline = VOGUEPipeline(
        case_dir=args.case_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        utils_root=args.utils_root,
        device=args.device,
        detect_stride=args.detect_stride,
        first_push_ratio=args.first_push_ratio,
        cut_hold_seconds=args.cut_hold_seconds,
        subject_fill=args.subject_fill,
        upper_fill=args.upper_fill,
        upper_body_ratio=args.upper_body_ratio,
    )
    pipeline.detect_all()
    output = pipeline.render(output_name=args.output_name)

    print("\n[VOGUE] Done.")
    print(f"[VOGUE] Video: {output}")
    print(f"[VOGUE] Detection JSON: {pipeline.output_dir / 'detection'}")
    print(f"[VOGUE] Three synchronized frames: {pipeline.output_dir / 'image'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
