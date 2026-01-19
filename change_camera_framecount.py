#!/usr/bin/env python3
"""
Resample a camera poses JSON to an exact target frame count while preserving temporal mapping.

The script expects a JSON with a top-level `poses` array where each pose is a 4x4 matrix
(list of 4 lists of 4 numbers). It linearly interpolates translation and spherically
interpolates rotation (via quaternion slerp) to produce exactly M poses.

Usage:
  python3 change_camera_framecount.py -i cameras.json -o cameras_out.json -m 81
"""

import argparse
import json
import math
import os
import sys
import numpy as np


def mat4_to_rt(mat):
    R = np.array([[mat[i][j] for j in range(3)] for i in range(3)], dtype=float)
    t = np.array([mat[i][3] for i in range(3)], dtype=float)
    return R, t


def rt_to_mat4(R, t):
    M = [[float(R[i, j]) for j in range(3)] + [float(t[i])] for i in range(3)]
    M.append([0.0, 0.0, 0.0, 1.0])
    return M


def rot_to_quat(R):
    m = R
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def quat_to_rot(q):
    w, x, y, z = q
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.array([
        [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
    ], dtype=float)
    return R


def quat_slerp(q0, q1, t):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = math.acos(max(min(dot, 1.0), -1.0))
    theta = theta_0 * t
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    return q0 * math.cos(theta) + q2 * math.sin(theta)


def resample_poses(poses, M):
    N = len(poses)
    if N == 0:
        raise ValueError("No poses to resample")

    # convert to R, t, q arrays
    Rs = []
    ts = []
    qs = []
    for mat in poses:
        R, t = mat4_to_rt(mat)
        Rs.append(R)
        ts.append(t)
        qs.append(rot_to_quat(R))

    Rs = np.array(Rs)
    ts = np.array(ts)
    qs = np.array(qs)

    out = []
    for k in range(M):
        p = (k * N) / M
        i = int(math.floor(p))
        alpha = p - i

        if i < 0:
            out.append(poses[0])
        elif i >= N - 1:
            out.append(poses[-1])
        else:
            q = quat_slerp(qs[i], qs[i + 1], alpha)
            t = (1.0 - alpha) * ts[i] + alpha * ts[i + 1]
            R = quat_to_rot(q)
            out.append(rt_to_mat4(R, t))

    return out


def main():
    parser = argparse.ArgumentParser(description='Resample camera poses JSON to exact frame count')
    parser.add_argument('-i', '--input', required=True, help='Input cameras JSON')
    parser.add_argument('-o', '--output', required=True, help='Output cameras JSON')
    parser.add_argument('-m', '--frames', required=True, type=int, help='Target frame count')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    with open(args.input, 'r') as f:
        data = json.load(f)

    if 'poses' not in data:
        print("Input JSON does not contain 'poses' key", file=sys.stderr)
        sys.exit(3)

    poses = data['poses']
    M = int(args.frames)
    if M <= 0:
        print("Target frames must be > 0", file=sys.stderr)
        sys.exit(4)

    out_poses = resample_poses(poses, M)
    data_out = dict(data)
    data_out['poses'] = out_poses
    data_out['original_num_poses'] = len(poses)
    data_out['target_num_poses'] = M

    with open(args.output, 'w') as f:
        json.dump(data_out, f, indent=2)

    print(f"Wrote {len(out_poses)} poses to {args.output}")


if __name__ == '__main__':
    main()
