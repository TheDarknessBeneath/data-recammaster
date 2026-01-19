#!/usr/bin/env python3
"""
Resample a video's frames to an exact target frame count while preserving duration.

Algorithm:
- Load all frames, compute original frame count `N` and duration `T = N / fps`.
- For target frame index k in [0..M-1], compute source position p = k * (N / M).
- Interpolate between floor(p) and ceil(p) using linear blending (cv2.addWeighted).
- Write M frames with new fps = M / T so duration stays approximately the same.

Usage example:
  python3 change_video_framecount.py -i input.mp4 -o output.mp4 -m 300

Note: This uses simple frame blending for interpolation (no optical-flow).
"""

import argparse
import math
import sys
import cv2
import os


def resample_frames(input_path, output_path, target_frames, codec='mp4v'):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    fps_orig = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Fallback: read frames to count if metadata missing
    frames = []
    if frame_count <= 0:
        while True:
            ret, frm = cap.read()
            if not ret:
                break
            frames.append(frm)
        frame_count = len(frames)
        if frame_count == 0:
            raise RuntimeError("Input video contains no frames.")
        # estimate fps if unknown
        if fps_orig <= 0.0:
            fps_orig = 30.0
        cap.release()
    else:
        # read frames into memory
        while True:
            ret, frm = cap.read()
            if not ret:
                break
            frames.append(frm)
        cap.release()

    if width == 0 or height == 0:
        h, w = frames[0].shape[:2]
        width, height = w, h

    N = frame_count
    M = int(target_frames)
    if M <= 0:
        raise ValueError("target_frames must be > 0")

    duration = N / fps_orig if fps_orig > 0 else float(N)
    fps_new = M / duration if duration > 0 else fps_orig

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, float(fps_new), (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Cannot open output for writing: {output_path}")

    # Map each output index k -> fractional source position p
    for k in range(M):
        p = (k * N) / M
        i = int(math.floor(p))
        alpha = p - i

        if i < 0:
            frame = frames[0]
        elif i >= N - 1:
            frame = frames[-1]
        else:
            a = frames[i]
            b = frames[i + 1]
            # linear blend; alpha=0 -> a, alpha=1 -> b
            frame = cv2.addWeighted(a, 1.0 - alpha, b, alpha, 0)

        out.write(frame)

    out.release()


def main():
    parser = argparse.ArgumentParser(description="Change video frame count without changing duration.")
    parser.add_argument('-i', '--input', required=True, help='Input video path')
    parser.add_argument('-o', '--output', required=True, help='Output video path')
    parser.add_argument('-m', '--frames', required=True, type=int, help='Target frame count (exact)')
    parser.add_argument('--codec', default='mp4v', help='FourCC codec (default mp4v)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    try:
        resample_frames(args.input, args.output, args.frames, codec=args.codec)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {args.frames} frames to {args.output}")


if __name__ == '__main__':
    main()
