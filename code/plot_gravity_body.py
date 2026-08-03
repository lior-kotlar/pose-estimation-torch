"""
Visual check of the gravity ("down") vector stored in *_analysis_smoothed.h5.

Writes an interactive plotly HTML with two 3D views, sampled every k frames:

  left  - LAB frame. The fly's body triad (x_body red, y_body green, z_body
          blue) drawn at the centre of mass, plus the gravity vector rotated
          back from body axes into the lab (R_body->lab @ gravity_body). The
          centre-of-mass trajectory joins the sampled frames as one continuous
          line in chronological order. Correctness check: every magenta arrow
          must point straight down (-z), whatever the fly is doing.

  right - BODY frame. The same gravity vectors drawn from the origin against
          fixed body axes, with a continuous line through their tips in
          chronological order - the path gravity traces over the fly's body as
          it pitches and rolls.

The round-trip residual |R_body->lab @ gravity_body - (0,0,-1)| is printed and
shown in the title; it should be ~1e-16.

Usage:
    python code/plot_gravity_body.py <h5-or-directory> [-k 100] [--scale 1.0]

<h5-or-directory> may be an *_analysis_smoothed.h5 file, a directory holding
one, or a directory whose immediate sub-directories each hold one.
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SUFFIX = "_analysis_smoothed.h5"
# Matches the collision-renamed copies collect_analysis_h5.py produces too
# (<stem>__<parent>.h5), which don't end in SUFFIX.
PATTERN = "*analysis_smoothed*.h5"
# Derived from the h5's own name, not fixed: a folder collected by
# collect_analysis_h5.py holds every movie side by side, and a constant name
# would have each movie overwrite the previous one's plot.
OUT_SUFFIX = "_gravity_check.html"
GRAVITY_LAB = np.array([0.0, 0.0, -1.0])

# body axis -> (display name, colour)
AXES = [("x_body (forward)", "#d62728"),
        ("y_body (left)", "#2ca02c"),
        ("z_body (dorsal)", "#1f77b4")]
GRAVITY_COLOR = "#e377c2"


def load(h5, key):
    if key not in h5:
        return None
    ds = h5[key]
    return ds[()] if ds.shape == () else ds[:]


def read_movie(h5_path):
    """Pull the per-frame body frame, centre of mass and gravity vector."""
    with h5py.File(h5_path, "r") as h5:
        # points_3D (arrow scale) and frame_index (hover labels) are optional
        data = {k: load(h5, k) for k in
                ("x_body", "y_body", "z_body", "center_of_mass", "gravity_body",
                 "points_3D", "frame_index")}
    missing = [k for k in ("x_body", "y_body", "z_body", "center_of_mass") if data[k] is None]
    if missing:
        raise KeyError(f"{h5_path} is missing {', '.join(missing)}")

    if data["gravity_body"] is None:
        # Movies analysed before gravity_body was added: derive it so old runs
        # can still be checked, but say so - this is no longer verifying what
        # the pipeline wrote.
        print(f"  note: no gravity_body dataset (pre-dates the feature); "
              f"deriving it from the body axes", file=sys.stderr)
        body_axes = np.stack((data["x_body"], data["y_body"], data["z_body"]), axis=1)
        data["gravity_body"] = body_axes @ GRAVITY_LAB
    return data


def valid_frames(data):
    """Frames where the body frame is fully defined (y_body only exists between
    first_y_body_frame and end_frame; outside it the arrays are NaN or zero)."""
    stacked = np.stack((data["x_body"], data["y_body"], data["z_body"],
                        data["gravity_body"], data["center_of_mass"]), axis=1)
    finite = np.isfinite(stacked).all(axis=(1, 2))
    non_degenerate = (np.linalg.norm(data["y_body"], axis=1) > 0.5)
    return np.where(finite & non_degenerate)[0]


def arrow_length(data, frames):
    """A physically meaningful arrow length: the fly's body length, falling back
    to a fraction of the trajectory's extent."""
    pts = data["points_3D"]
    if pts is not None and pts.ndim == 3 and pts.shape[1] >= 2:
        body = np.linalg.norm(pts[frames, -1, :] - pts[frames, -2, :], axis=1)
        body = body[np.isfinite(body)]
        if body.size and np.median(body) > 0:
            return float(np.median(body))
    com = data["center_of_mass"][frames]
    diag = float(np.linalg.norm(com.max(axis=0) - com.min(axis=0)))
    return 0.08 * diag if diag > 0 else 1.0


def shafts(starts, vecs, name, color, width=4, legendgroup=None):
    """One trace holding every arrow shaft, split by None separators."""
    xs, ys, zs = [], [], []
    for s, v in zip(starts, vecs):
        e = s + v
        xs += [s[0], e[0], None]
        ys += [s[1], e[1], None]
        zs += [s[2], e[2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name,
                        legendgroup=legendgroup or name,
                        line=dict(color=color, width=width), hoverinfo="skip")


def heads(starts, vecs, name, color, size, legendgroup=None):
    """Cone arrowheads sitting at the arrow tips."""
    tips = starts + vecs
    return go.Cone(x=tips[:, 0], y=tips[:, 1], z=tips[:, 2],
                   u=vecs[:, 0], v=vecs[:, 1], w=vecs[:, 2],
                   sizemode="absolute", sizeref=size, anchor="tip",
                   colorscale=[[0, color], [1, color]], showscale=False,
                   name=name, legendgroup=legendgroup or name, showlegend=False,
                   hoverinfo="skip")


def plot_one(h5_path, k, scale, out_path=None):
    print(f"{os.path.basename(h5_path)}", flush=True)
    data = read_movie(h5_path)
    frames = valid_frames(data)
    if frames.size == 0:
        print("  no frames with a fully defined body frame - nothing to plot",
              file=sys.stderr)
        return False

    sampled = frames[::k]
    xb, yb, zb = (data[a][sampled] for a in ("x_body", "y_body", "z_body"))
    g_body = data["gravity_body"][sampled]
    com = data["center_of_mass"][sampled]

    # Round-trip: rotate the stored body-frame vector back into the lab. If the
    # convention is right this is (0, 0, -1) on every frame.
    g_all = data["gravity_body"][frames]
    axes_all = np.stack((data["x_body"][frames], data["y_body"][frames],
                         data["z_body"][frames]), axis=1)      # rows = R_lab->body
    g_lab_all = np.einsum("fij,fi->fj", axes_all, g_all)        # R^T @ g_body
    residual = float(np.abs(g_lab_all - GRAVITY_LAB).max())
    norms = np.linalg.norm(g_all, axis=1)
    print(f"  frames with a body frame: {frames.size} "
          f"(plotting every {k} -> {sampled.size})")
    print(f"  |gravity_body| in [{norms.min():.6f}, {norms.max():.6f}]")
    print(f"  max |R_body->lab @ gravity_body - (0,0,-1)| = {residual:.3e}")

    g_lab = np.einsum("fij,fi->fj", np.stack((xb, yb, zb), axis=1), g_body)

    arrow = arrow_length(data, frames) * scale
    head_size = 0.25 * arrow

    # frame labels: trigger-relative if the movie carries them
    frame_index = data["frame_index"]
    labels = (frame_index[sampled] if frame_index is not None
              and len(frame_index) > sampled.max() else sampled)
    label_name = "trigger frame" if frame_index is not None else "array index"

    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("lab frame: body triad + gravity along the trajectory",
                        "body frame: gravity seen by the fly"))

    # ---------------- left: lab frame ----------------
    # continuous chronological line through every valid frame, coloured by time
    traj = data["center_of_mass"][frames]
    fig.add_trace(go.Scatter3d(
        x=traj[:, 0], y=traj[:, 1], z=traj[:, 2], mode="lines",
        name="centre of mass (time order)",
        line=dict(color=frames, colorscale="Viridis", width=4),
        hoverinfo="skip"), row=1, col=1)

    for vecs, (name, color) in zip((xb, yb, zb), AXES):
        fig.add_trace(shafts(com, vecs * arrow, name, color), row=1, col=1)
        fig.add_trace(heads(com, vecs * arrow, name, color, head_size), row=1, col=1)
    fig.add_trace(shafts(com, g_lab * arrow, "gravity", GRAVITY_COLOR, width=6),
                  row=1, col=1)
    fig.add_trace(heads(com, g_lab * arrow, "gravity", GRAVITY_COLOR, head_size),
                  row=1, col=1)

    # sampled positions, hoverable so a suspicious arrow can be identified
    fig.add_trace(go.Scatter3d(
        x=com[:, 0], y=com[:, 1], z=com[:, 2], mode="markers",
        marker=dict(size=3, color="black"), name="sampled frames",
        customdata=labels, hovertemplate=f"{label_name} %{{customdata}}<extra></extra>"),
        row=1, col=1)

    # ---------------- right: body frame ----------------
    origin = np.zeros((len(g_body), 3))
    unit = np.eye(3)
    for i, (name, color) in enumerate(AXES):
        fig.add_trace(shafts(np.zeros((1, 3)), unit[i][None, :], name, color,
                             width=6, legendgroup=name), row=1, col=2)
        fig.add_trace(heads(np.zeros((1, 3)), unit[i][None, :], name, color,
                            0.12, legendgroup=name), row=1, col=2)
    fig.add_trace(shafts(origin, g_body, "gravity", GRAVITY_COLOR, width=2),
                  row=1, col=2)
    # continuous line through the tips, in chronological order
    tips_all = data["gravity_body"][frames]
    fig.add_trace(go.Scatter3d(
        x=tips_all[:, 0], y=tips_all[:, 1], z=tips_all[:, 2], mode="lines",
        name="gravity tip (time order)",
        line=dict(color=frames, colorscale="Viridis", width=5),
        hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter3d(
        x=g_body[:, 0], y=g_body[:, 1], z=g_body[:, 2], mode="markers",
        marker=dict(size=3, color="black"), name="sampled frames",
        showlegend=False, customdata=labels,
        hovertemplate=(f"{label_name} %{{customdata}}<br>"
                       "gravity_body (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>")),
        row=1, col=2)

    axis_style = dict(showbackground=True, backgroundcolor="rgb(245,245,250)")
    fig.update_layout(
        title=(f"{os.path.basename(h5_path)} &mdash; every {k}th frame &mdash; "
               f"max |R@g_body - (0,0,-1)| = {residual:.1e}"),
        scene=dict(aspectmode="data",
                   xaxis=dict(title="lab x", **axis_style),
                   yaxis=dict(title="lab y", **axis_style),
                   zaxis=dict(title="lab z (up)", **axis_style)),
        scene2=dict(aspectmode="cube",
                    xaxis=dict(title="x_body", range=[-1.2, 1.2], **axis_style),
                    yaxis=dict(title="y_body", range=[-1.2, 1.2], **axis_style),
                    zaxis=dict(title="z_body", range=[-1.2, 1.2], **axis_style)),
        legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=60, b=0))

    out_path = out_path or os.path.join(
        os.path.dirname(h5_path),
        os.path.splitext(os.path.basename(h5_path))[0] + OUT_SUFFIX)
    fig.write_html(out_path)
    print(f"  wrote {out_path}")
    return True


def find_h5(directory):
    return sorted(glob.glob(os.path.join(directory, PATTERN)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help=f"an *{SUFFIX} file, a directory holding one, "
                                   f"or a directory of such sub-directories")
    ap.add_argument("-k", "--every", type=int, default=100, metavar="K",
                    help="draw the body frame every K frames (default: 100)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply the arrow length (default 1 = one body length)")
    ap.add_argument("--out", default=None,
                    help=f"output HTML path (single-file mode only; "
                         f"default <h5 name>{OUT_SUFFIX} next to the h5)")
    args = ap.parse_args()

    if args.every < 1:
        sys.exit("-k/--every must be >= 1")

    if os.path.isfile(args.target):
        sys.exit(0 if plot_one(args.target, args.every, args.scale, args.out) else 1)

    if not os.path.isdir(args.target):
        sys.exit(f"No such file or directory: {args.target}")

    direct = find_h5(args.target)
    if len(direct) == 1:
        sys.exit(0 if plot_one(direct[0], args.every, args.scale, args.out) else 1)
    if len(direct) > 1:
        sys.exit(f"Found {len(direct)} *{SUFFIX} files directly in {args.target}; "
                 f"name one explicitly.")

    subdirs = sorted(d for d in (os.path.join(args.target, name)
                                 for name in os.listdir(args.target))
                     if os.path.isdir(d))
    n_ok = 0
    for sub in subdirs:
        matches = find_h5(sub)
        if len(matches) != 1:
            continue
        try:
            n_ok += bool(plot_one(matches[0], args.every, args.scale))
        except Exception as e:
            print(f"  {sub}: {e}", file=sys.stderr)
    if n_ok == 0:
        sys.exit(f"No *{SUFFIX} files found under {args.target}")
    print(f"done: {n_ok} plotted")


if __name__ == "__main__":
    main()
