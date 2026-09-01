"""
Interactive per-movie flight viewer from a *_analysis_smoothed.h5 file.

Writes one self-contained HTML file next to the h5:

  <stem>_flight_viewer.html

The page has two sections side by side, and one control bar along the bottom:

  right   the fly itself -- the same 3D content as the right-hand panel of the
          prediction mp4 (18-joint skeleton, body triad, wing span and chord
          vectors, stroke plane), but flying through the lab frame rather than
          pinned inside a camera that follows the centre of mass, and
          scrubbable rather than rendered
  left    three graph panels, each with its own menu of everything the analysis
          h5 holds. A panel shows either a time series against the
          trigger-relative x-axis the mp4 counter, the analysis CSV and
          wing_angles.png all share, or a 3D plot of one wing's path through
          (phi, theta, psi) coloured by time
  bottom  the fader, play/pause and slow-motion speed, the lab/follow view
          switch and the frames/ms axis switch

Everything stays tied to one instant: the fader marks it in every panel, and
zooming one time series zooms the others and narrows the stretch of the
angle-space trajectory on show. Hovering a panel marks that instant on the
other panels but deliberately does NOT move the fly -- running the pointer
across a graph should not drag the animation around with it. Clicking does
seek.

Why this is hand-rolled rather than a plotly animation: a `go.Frame` per movie
frame duplicates every trace, which at ~4500 frames is a several-hundred-MB
file (see Visualizer.visualize_analysis_3D_html, which is capped at 200 frames
for exactly that reason). Here each array is embedded ONCE as base64 float32
and the animation is a few Plotly.restyle calls driven from JavaScript, which
puts a full-length movie at roughly 3 MB of data plus the 4.8 MB of inlined
plotly.js.

Usage:
    python code/plot_flight_viewer.py <target>

<target> may be an *_analysis_smoothed.h5, a directory containing exactly one,
or a directory whose immediate sub-directories each contain one. The mode is
detected automatically.
"""
import argparse
import base64
import glob
import json
import os
import sys

import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

# frame_axis is what makes this viewer share one x-axis with the mp4 counter,
# the analysis CSV and the validation PNGs; `load` is the tolerant read that
# lets an older h5 lose a dataset without taking the whole viewer down.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_wing_and_body import load, frame_axis, x_axis  # noqa: E402

SUFFIX = "_analysis_smoothed.h5"
# Also matches collect_analysis_h5.py's <stem>__<parent>.h5 renaming.
PATTERN = "*analysis_smoothed*.h5"
OUT_SUFFIX = "_flight_viewer.html"

# The CDN build must be the same plotly.js the installed plotly python ships,
# or a figure built here can use an attribute the served bundle does not know.
PLOTLY_CDN = "https://cdn.plot.ly/plotly-{version}.min.js"

# The 18 joints: 0-6 left wing outline, 7 left hinge, 8-14 right wing outline,
# 15 right hinge, 16 tail, 17 head. Copied verbatim from the mp4 renderer
# (Visualizer.create_movie_mp4) so the two draw the same fly.
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6),
               (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (8, 14),
               (16, 17)]
# The hinges sit inside the body and only clutter the scatter; the mp4 hides
# them too.
HIDDEN_JOINTS = (7, 15)
VISIBLE_JOINTS = [i for i in range(18) if i not in HIDDEN_JOINTS]

# Half-width of the cube the "Follow" view keeps around the centre of mass.
# The mp4 uses the same 3 mm (Visualizer.create_movie_mp4's zoom_scale).
FOLLOW_HALF_WIDTH = 0.003
# Edge of the square drawn for the stroke plane, as in the mp4.
STROKE_PLANE_SIZE = 0.005

# (group, label, [(h5 key, column or None, trace name)], y-axis unit, scale)
#
# Three things this table has to get right, all of them traps in the h5:
#   * wing_tips_speed stores RIGHT in column 0 and LEFT in column 1 -- the
#     reverse of every other left/right pair in the file (get_wing_tips_speed
#     concatenates right first). Hence the column numbers below.
#   * center_of_mass is in metres; the analysis CSV publishes it in mm, so the
#     scale factor keeps the two agreeing.
#   * anything whose length differs from num_frames is dropped at build time
#     (see build_signals) -- roll_angle_roni is stored unpadded and would
#     silently misalign the shared x-axis.
SIGNALS = [
    ("Wing angles", "phi (stroke)",
     [("wings_phi_left", None, "left"), ("wings_phi_right", None, "right")], "deg", 1.0),
    ("Wing angles", "theta (deviation)",
     [("wings_theta_left", None, "left"), ("wings_theta_right", None, "right")], "deg", 1.0),
    ("Wing angles", "psi (rotation)",
     [("wings_psi_left", None, "left"), ("wings_psi_right", None, "right")], "deg", 1.0),
    ("Wing angles", "phi rate",
     [("wings_phi_left_dot", None, "left"), ("wings_phi_right_dot", None, "right")], "deg/s", 1.0),
    ("Wing angles", "theta rate",
     [("wings_theta_left_dot", None, "left"), ("wings_theta_right_dot", None, "right")], "deg/s", 1.0),
    ("Wing angles", "psi rate",
     [("wings_psi_left_dot", None, "left"), ("wings_psi_right_dot", None, "right")], "deg/s", 1.0),
    ("Wing angles", "deformation angle",
     [("left_deformation_angle", None, "left"), ("right_deformation_angle", None, "right")], "deg", 1.0),

    ("Body attitude", "yaw / pitch / roll",
     [("yaw_angle", None, "yaw"), ("pitch_angle", None, "pitch"),
      ("roll_angle", None, "roll")], "deg", 1.0),
    ("Body attitude", "yaw / pitch / roll rate",
     [("yaw_dot", None, "yaw rate"), ("pitch_dot", None, "pitch rate"),
      ("roll_dot", None, "roll rate")], "deg/s", 1.0),
    ("Body attitude", "body rates p, q, r",
     [("p", None, "p (roll)"), ("q", None, "q (pitch)"), ("r", None, "r (yaw)")], "deg/s", 1.0),
    ("Body attitude", "omega_body",
     [("omega_body", 0, "about x_body"), ("omega_body", 1, "about y_body"),
      ("omega_body", 2, "about z_body")], "deg/s", 1.0),
    ("Body attitude", "omega_lab",
     [("omega_lab", 0, "x"), ("omega_lab", 1, "y"), ("omega_lab", 2, "z")], "deg/s", 1.0),
    ("Body attitude", "angular speed",
     [("angular_speed_body", None, "|omega_body|")], "deg/s", 1.0),
    ("Body attitude", "angular acceleration (components)",
     [("omega_body_dot", 0, "about x_body (roll)"), ("omega_body_dot", 1, "about y_body (pitch)"),
      ("omega_body_dot", 2, "about z_body (yaw)")], "10^3 deg/s^2", 1e-3),
    ("Body attitude", "angular acceleration (magnitude)",
     [("angular_acceleration_body", None, "|omega_body_dot|")], "10^3 deg/s^2", 1e-3),

    ("Translation", "CM position",
     [("center_of_mass", 0, "x"), ("center_of_mass", 1, "y"),
      ("center_of_mass", 2, "z")], "mm", 1e3),
    ("Translation", "CM velocity",
     [("CM_dot", 0, "x"), ("CM_dot", 1, "y"), ("CM_dot", 2, "z")], "m/s", 1.0),
    ("Translation", "CM speed", [("CM_speed", None, "|v|")], "m/s", 1.0),
    ("Translation", "wing tip speed",
     [("wing_tips_speed", 1, "left"), ("wing_tips_speed", 0, "right")], "m/s", 1.0),
    ("Translation", "gravity in body axes",
     [("gravity_body", 0, "x_body"), ("gravity_body", 1, "y_body"),
      ("gravity_body", 2, "z_body")], "unit", 1.0),
]

# The angle-space plots are menu entries like any other, so any row can show
# one. They are 3D scenes rather than time series, which is why a row is its
# own figure -- see build_row_template.
WING_SPACE = [("Wing angles", "wing angle space (left)", "left"),
              ("Wing angles", "wing angle space (right)", "right")]

# What the three rows show when the file is first opened. Row 3 starts on an
# angle-space plot so the two kinds are both visible without touching a menu.
DEFAULT_ROWS = ["phi (stroke)", "angular acceleration (components)",
                "wing angle space (left)"]

# Slot colours. Left/right pairs land on blue/orange, matching wing_angles.png;
# xyz triples get blue/orange/green.
SLOT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
MAX_TRACES_PER_ROW = 3
N_ROWS = 3


# --------------------------------------------------------------------------
# payload packing
# --------------------------------------------------------------------------

def pack(arr):
    """A float32 array as base64, for embedding once instead of per frame.

    JSON'ing points_3D costs 2.9 MB where the raw float32 bytes cost 1.3 MB,
    and the browser hands a Float32Array straight to Plotly without parsing.
    NaN survives the round trip, which is what keeps the unanalysed head of
    every series drawn as a gap rather than as zero.
    """
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    return {"b": base64.b64encode(a.tobytes()).decode("ascii"), "shape": list(a.shape)}


def as_text(value, default=""):
    """h5 scalar strings arrive as bytes, numpy bytes, or object arrays."""
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else (value[0] if len(value) else default)
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", "replace")
    return str(value)


def stroke_plane_corners(cm, normal, y_body, size=STROKE_PLANE_SIZE):
    """The square drawn for the stroke plane, one per frame.

    Same construction as the mp4's compute_plane_data: centred on the body's
    centre of mass and spanned by y_body and (normal x y_body). Frames where
    y_body is undefined come out NaN, which hides the patch rather than drawing
    it somewhere arbitrary.
    """
    d = size / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        yb = y_body / np.linalg.norm(y_body, axis=1, keepdims=True)
        u = np.cross(normal, yb)
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
    return np.stack([cm + d * (u + yb), cm + d * (u - yb),
                     cm + d * (-u - yb), cm + d * (-u + yb)], axis=1)


def build_geometry(h5, step):
    """Everything the 3D scene moves, decimated by `step`.

    Returns (packed dict, n_frames, scales). Missing datasets become NaN arrays
    of the right shape so the browser still has something to restyle -- an h5
    written before a given vector existed loses that arrow, not the viewer.
    """
    points = np.asarray(load(h5, "points_3D"), dtype=float)[::step]
    n = len(points)

    def vec(key, cols=3):
        v = load(h5, key)
        if v is None:
            return np.full((n, cols), np.nan)
        v = np.asarray(v, dtype=float)[::step]
        return v[:n] if len(v) >= n else np.full((n, cols), np.nan)

    cm = vec("center_of_mass")
    x_body, y_body, z_body = vec("x_body"), vec("y_body"), vec("z_body")
    stroke = load(h5, "stroke_planes")
    # stroke_planes is [nx, ny, nz, d]; only the normal is wanted.
    normal = (np.asarray(stroke, dtype=float)[::step][:n, :3]
              if stroke is not None else np.full((n, 3), np.nan))

    # Arrow lengths follow the fly's own size rather than a fixed millimetre,
    # so the same script suits a different rig without retuning.
    body_len = np.nanmean(np.linalg.norm(points[:, 17] - points[:, 16], axis=1))
    if not np.isfinite(body_len) or body_len <= 0:
        body_len = 0.0024
    tip_reach = np.nanmean(np.linalg.norm(points[:, 2] - points[:, 6], axis=1))
    if not np.isfinite(tip_reach) or tip_reach <= 0:
        tip_reach = body_len

    raw = {
        "points": points, "cm": cm,
        "x_body": x_body, "y_body": y_body, "z_body": z_body,
        "left_wing_CM": vec("left_wing_CM"), "right_wing_CM": vec("right_wing_CM"),
        "left_wing_span": vec("left_wing_span"), "right_wing_span": vec("right_wing_span"),
        "left_wing_chord": vec("left_wing_chord"), "right_wing_chord": vec("right_wing_chord"),
        "stroke_corners": stroke_plane_corners(cm, normal, y_body),
    }
    packed = {k: pack(v) for k, v in raw.items()}
    scales = {"body_vec": 0.6 * body_len, "wing_vec": tip_reach,
              "follow": FOLLOW_HALF_WIDTH}
    return raw, packed, n, scales


def build_signals(h5, n, step, n_full):
    """The signal registry, resolved against this file and packed.

    An entry survives only if every one of its series is present AND still the
    right length after decimation. The length test is the important half:
    roll_angle_roni is stored unpadded (it covers only the frames where y_body
    is defined), so plotting it against the shared x-axis would shift every
    sample. The test is applied AFTER the [::step] slice, because a short array
    decimated by K is short by K times as much and would otherwise slip past a
    check made on the raw length.
    """
    out, dropped = [], []
    for group, label, members, unit, scale in SIGNALS:
        traces, ok = [], True
        for key, col, name in members:
            raw = load(h5, key)
            if raw is None:
                ok = False
                dropped.append(f"{label}: {key} missing")
                break
            arr = np.asarray(raw, dtype=float)
            n_raw = len(arr)
            arr = arr[::step][:n]
            if len(arr) != n:
                ok = False
                dropped.append(f"{label}: {key} has {n_raw} rows, expected {n_full}")
                break
            if col is not None:
                if arr.ndim < 2 or arr.shape[1] <= col:
                    ok = False
                    dropped.append(f"{label}: {key} has no column {col}")
                    break
                arr = arr[:, col]
            traces.append({"name": name, "y": pack(arr * scale)})
        if ok:
            out.append({"group": group, "label": label, "unit": unit,
                        "kind": "2d", "traces": traces})
    for group, label, side in WING_SPACE:
        out.append({"group": group, "label": label, "unit": "deg",
                    "kind": "ws", "side": side, "traces": []})
    return out, dropped


def build_perturbation(h5, frames):
    """The declared perturbation window, in the same terms the mp4 counter uses.

    `onset` is trigger-relative and always known once an experiment is declared
    perturbed; the duration often is not, and an unknown end stays open-ended
    rather than being guessed at (mirroring plot_wing_and_body.mark_perturbation).
    An onset outside the built range is normal -- the prescan picks its range
    from fly visibility and knows nothing about the perturbation -- so the band
    is clipped to the axis instead of being dropped.
    """
    if not load(h5, "perturbation"):
        return None
    onset = load(h5, "perturbation_start_frame")
    if onset is None:
        return None
    end_known = bool(load(h5, "perturbation_end_known"))
    end = load(h5, "perturbation_end_frame") if end_known else None
    label = as_text(load(h5, "perturbation_type"), "perturbation")
    if not end_known:
        label += " (end unrecorded)"
    lo, hi = float(frames[0]), float(frames[-1])
    onset = float(onset)
    return {
        "onset": onset,
        "end": float(end) if end is not None else None,
        "end_known": end_known,
        "label": label,
        # Band drawn on the axis, clipped; None when it misses the range entirely.
        "band": [max(onset, lo), min(float(end) if end is not None else hi, hi)]
                if min(float(end) if end is not None else hi, hi) > max(onset, lo) else None,
        "onset_visible": lo <= onset <= hi,
    }


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def joint_colors():
    """The mp4's per-joint HSV rainbow, so a joint keeps its colour across the
    two views. Computed here rather than pulled from matplotlib, which this
    script otherwise has no reason to import."""
    import colorsys
    return ["rgb(%d,%d,%d)" % tuple(round(255 * c) for c in
                                    colorsys.hsv_to_rgb(h, 1.0, 1.0))
            for h in np.linspace(0, 1, 18, endpoint=False)]


def skeleton_coords(pts):
    """One polyline per frame for all 15 bones: a, b, NaN, a, b, NaN, ...

    Drawing the skeleton as a single NaN-separated trace instead of 15 traces
    turns the per-frame update into one restyle entry.
    """
    out = np.full((len(CONNECTIONS) * 3, 3), np.nan)
    for k, (i, j) in enumerate(CONNECTIONS):
        out[3 * k] = pts[i]
        out[3 * k + 1] = pts[j]
    return out


def seg(a, b):
    return dict(x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]])


def build_scene_figure(raw, scales):
    """The 3D lab view. Trace order here is the contract the JS restyle relies
    on -- SCENE_DYNAMIC in the control layer indexes into it."""
    pts0 = raw["points"][0]
    cm = raw["cm"]
    sk = skeleton_coords(pts0)
    colors = joint_colors()
    bv, wv = scales["body_vec"], scales["wing_vec"]

    def arrow(origin, vec_, length):
        return seg(origin[0], origin[0] + vec_[0] * length)

    traces = [
        # 0: the whole flight path, drawn once and never touched again -- it is
        # what makes the lab view read as a trajectory rather than a hovering fly.
        go.Scatter3d(x=cm[:, 0], y=cm[:, 1], z=cm[:, 2], mode="lines",
                     line=dict(color="rgba(120,120,120,0.45)", width=2),
                     name="flight path", hoverinfo="skip"),
        # 1: recent history, brighter, so direction of travel is legible.
        go.Scatter3d(x=[], y=[], z=[], mode="lines",
                     line=dict(color="#d62728", width=4), name="recent path",
                     hoverinfo="skip"),
        # 2
        go.Scatter3d(x=sk[:, 0], y=sk[:, 1], z=sk[:, 2], mode="lines",
                     line=dict(color="black", width=3), name="skeleton",
                     hoverinfo="skip"),
        # 3
        go.Scatter3d(x=pts0[VISIBLE_JOINTS, 0], y=pts0[VISIBLE_JOINTS, 1],
                     z=pts0[VISIBLE_JOINTS, 2], mode="markers",
                     marker=dict(size=4, color=[colors[i] for i in VISIBLE_JOINTS]),
                     name="joints", hoverinfo="skip"),
        # 4-6: the body triad. The mp4 draws all three red; splitting the colours
        # the way plot_gravity_body does makes the attitude readable at a glance.
        go.Scatter3d(**arrow(raw["cm"], raw["x_body"], bv), mode="lines",
                     line=dict(color="#d62728", width=6), name="x_body (forward)"),
        go.Scatter3d(**arrow(raw["cm"], raw["y_body"], bv), mode="lines",
                     line=dict(color="#2ca02c", width=6), name="y_body (left)"),
        go.Scatter3d(**arrow(raw["cm"], raw["z_body"], bv), mode="lines",
                     line=dict(color="#1f77b4", width=6), name="z_body (dorsal)"),
        # 7-10: span and chord, from each wing's own centre of mass.
        go.Scatter3d(**arrow(raw["left_wing_CM"], raw["left_wing_span"], wv), mode="lines",
                     line=dict(color="#e377c2", width=5), name="left span"),
        go.Scatter3d(**arrow(raw["left_wing_CM"], raw["left_wing_chord"], wv), mode="lines",
                     line=dict(color="#9467bd", width=5), name="left chord"),
        go.Scatter3d(**arrow(raw["right_wing_CM"], raw["right_wing_span"], wv), mode="lines",
                     line=dict(color="#e377c2", width=5), name="right span",
                     showlegend=False),
        go.Scatter3d(**arrow(raw["right_wing_CM"], raw["right_wing_chord"], wv), mode="lines",
                     line=dict(color="#9467bd", width=5), name="right chord",
                     showlegend=False),
        # 11
        go.Mesh3d(x=raw["stroke_corners"][0][:, 0], y=raw["stroke_corners"][0][:, 1],
                  z=raw["stroke_corners"][0][:, 2], i=[0, 0], j=[1, 2], k=[2, 3],
                  color="#2ca02c", opacity=0.25, name="stroke plane",
                  showlegend=True, hoverinfo="skip"),
        # 12
        go.Scatter3d(x=[cm[0, 0]], y=[cm[0, 1]], z=[cm[0, 2]], mode="markers",
                     marker=dict(size=5, color="#ff7f0e"), name="centre of mass"),
    ]

    # Lab view: one fixed box around everything the scene ever draws, over the
    # whole movie. Not just the centre of mass, and not just the joints either:
    # the stroke-plane square and the arrow tips both reach past the outermost
    # joint, and an axis range that excludes them would clip them at the wall.
    corners = [raw["points"].reshape(-1, 3),
               raw["stroke_corners"].reshape(-1, 3)]
    for origin, vec in [("cm", "x_body"), ("cm", "y_body"), ("cm", "z_body")]:
        corners.append(raw[origin] + raw[vec] * bv)
    for side in ("left", "right"):
        for vec in ("span", "chord"):
            corners.append(raw[f"{side}_wing_CM"] + raw[f"{side}_wing_{vec}"] * wv)
    everything = np.concatenate(corners, axis=0)
    lo = np.nanmin(everything, axis=0)
    hi = np.nanmax(everything, axis=0)
    pad = 0.25 * scales["body_vec"]
    lo, hi = lo - pad, hi + pad

    # aspectmode MUST be "manual" here, and the ratio must be proportional to
    # those ranges. Under "data" -- the obvious choice -- plotly.js derives the
    # ratio from the bounding box of the trace data on every single replot:
    #
    #     for (s=0; s<3; ++s) d[s] = 1 / (h[1][s] - h[0][s]);   // trace span
    #     Z[o] = Math.pow(H.acc, 1/H.count) / d[o];             // => S[o]/(S0*S1*S2)^(1/3)
    #
    # Restyling the fly to a new frame changes that box -- the wings fold and
    # extend -- so the scene's proportions would be recomputed every frame and
    # the whole plot would visibly pulse at the wingbeat frequency. Pinning the
    # ratio to the (fixed) ranges stops the pulsing AND is what makes the axes
    # equal: one metre is then the same number of pixels on all three.
    span = hi - lo
    aspect = (span / span.max()).tolist()

    axis = dict(nticks=8, showspikes=False, tickformat=".4f")
    fig = go.Figure(traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x (m)", range=[lo[0], hi[0]], **axis),
            yaxis=dict(title="y (m)", range=[lo[1], hi[1]], **axis),
            zaxis=dict(title="z (m)", range=[lo[2], hi[2]], **axis),
            aspectmode="manual",
            aspectratio=dict(x=aspect[0], y=aspect[1], z=aspect[2]),
            dragmode="orbit",
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.7)", font=dict(size=11)),
        uirevision="scene",  # keep the user's orbit across every restyle
    )
    return fig, [lo.tolist(), hi.tolist()], aspect

def build_row_template(x_label, pert):
    """One graph row, empty. All three rows are newPlot'd from this same spec.

    Each row is its own figure rather than a row of one shared subplot, because
    a row must be able to hold a 3D angle-space scene as readily as a 2D time
    series, and a Scatter3d needs a scene rather than xy axes. The x-linking
    that make_subplots gave for free is re-created in the control layer, which
    is a dozen lines and buys the ability to mix the two kinds.

    Shape indices are a contract with the control layer: 0 is the perturbation
    band and 1 its onset line. Both are always present, merely invisible when
    the movie declares no perturbation, so the indices never shift.
    """
    fig = go.Figure([
        go.Scattergl(x=[], y=[], mode="lines", line=dict(color=SLOT_COLORS[t], width=1),
                     name="", visible=False,
                     hovertemplate="%{y:.4g}<extra>%{fullData.name}</extra>")
        for t in range(MAX_TRACES_PER_ROW)])
    fig.update_layout(
        # Only the perturbation band lives here; it is static, so drawing it as
        # a shape costs nothing. The time cursor is a DOM overlay instead --
        # see the .cursor rule in the stylesheet.
        shapes=[
            dict(type="rect", xref="x", yref="y domain",
                 x0=(pert["band"][0] if pert and pert["band"] else 0),
                 x1=(pert["band"][1] if pert and pert["band"] else 0), y0=0, y1=1,
                 fillcolor="rgba(214,39,40,0.08)", line_width=0, layer="below",
                 visible=bool(pert and pert["band"])),
            dict(type="line", xref="x", yref="y domain",
                 x0=(pert["onset"] if pert else 0), x1=(pert["onset"] if pert else 0),
                 y0=0, y1=1, line=dict(color="rgba(214,39,40,0.6)", width=1),
                 layer="below", visible=bool(pert and pert["onset_visible"])),
        ],
        xaxis=dict(title=x_label), yaxis=dict(title=""),
        hovermode="x", dragmode="zoom",
        margin=dict(l=58, r=14, t=8, b=34),
        legend=dict(orientation="h", y=1.14, x=0, font=dict(size=10)),
        showlegend=True,
    )
    return fig


def build_ws_template(side, unit_label):
    """A wing's path through (phi, theta, psi), empty.

    Filled by the control layer from the packed angles, so that the zoom link
    can re-slice it without depending on plotly.js having normalised the
    base64 {dtype, bdata} spec plotly.py emits for a numpy array.

    Trace 0 is the trajectory, trace 1 the marker for the current frame, and
    trace 2 the marker for whatever the pointer is over.
    """
    fig = go.Figure([
        go.Scatter3d(x=[], y=[], z=[], mode="lines+markers",
                     line=dict(color="rgba(120,120,120,0.35)", width=1),
                     marker=dict(size=1.8, colorscale="Viridis", showscale=True,
                                 colorbar=dict(title=dict(text=unit_label, side="right"),
                                               thickness=9, len=0.75, x=1.0,
                                               tickfont=dict(size=9))),
                     name=side,
                     hovertemplate=("phi %{x:.1f}<br>theta %{y:.1f}"
                                    "<br>psi %{z:.1f}<extra></extra>")),
        go.Scatter3d(x=[], y=[], z=[], mode="markers",
                     marker=dict(size=7, color="#d62728"), name="now",
                     hoverinfo="skip"),
        # Where the pointer is, as opposed to where the fader is. Hovering must
        # not move the fly, so the two markers are separate.
        go.Scatter3d(x=[], y=[], z=[], mode="markers",
                     marker=dict(size=6, color="#1f77b4", symbol="diamond"),
                     name="hover", hoverinfo="skip"),
    ])
    fig.update_layout(
        scene=dict(xaxis_title="phi", yaxis_title="theta", zaxis_title="psi",
                   camera=dict(eye=dict(x=1.5, y=1.5, z=1.1))),
        margin=dict(l=0, r=0, t=4, b=0), showlegend=False,
    )
    return fig


def read_wing_angles(h5, side, n, step):
    """(phi, theta, psi) for one wing, decimated to match everything else."""
    out = {}
    for name in ("phi", "theta", "psi"):
        v = load(h5, f"wings_{name}_{side}")
        out[name] = (np.asarray(v, dtype=float)[::step][:n] if v is not None
                     else np.full(n, np.nan))
    return out


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------

# Tokens rather than an f-string or str.Template: the JavaScript below is full
# of { } blocks and ${ } template literals, either of which would collide.
HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>@@TITLE@@</title>
<style>
  html,body{margin:0;padding:0;height:100%;font:13px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#222;background:#fff}
  #page{display:flex;flex-direction:column;height:100%}
  #head{padding:6px 12px;border-bottom:1px solid #ddd;font-size:12px;color:#555;flex:0 0 auto}
  #head b{color:#111;font-size:13px}
  #main{display:flex;flex:1 1 auto;min-height:0}
  #graphsec{flex:1 1 52%;display:flex;flex-direction:column;min-width:0;min-height:0}
  .row{flex:1 1 33.33%;display:flex;flex-direction:column;min-height:0;border-bottom:1px solid #f0f0f0;position:relative}
  /* The time cursor and the hover marker are plain DOM, not plotly shapes:
     moving a shape costs a full redraw of the panel's 4500-point trace, which
     is far too much to do on every animation frame. */
  .cursor,.hline{position:absolute;width:0;pointer-events:none;display:none;z-index:5}
  .cursor{border-left:1.5px solid #d62728}
  .hline{border-left:1.5px dashed #1f77b4}
  .rowhead{flex:0 0 auto;padding:3px 8px 0}
  .rowhead select{font:12px inherit;max-width:100%}
  .rowplot{flex:1 1 auto;min-height:0}
  #scene3d{flex:1 1 48%;min-width:0;border-left:1px solid #ddd}
  #bar{flex:0 0 auto;padding:7px 12px;border-top:1px solid #ccc;background:#fafafa;
       display:flex;align-items:center;gap:12px;flex-wrap:wrap}
  #bar button{font:12px inherit;padding:3px 9px;cursor:pointer;border:1px solid #bbb;background:#fff;border-radius:3px}
  #bar button:hover{background:#eee}
  #play{min-width:56px;font-weight:600}
  #fader{flex:1 1 300px;min-width:180px}
  #read{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;white-space:pre;min-width:230px}
  #rate{color:#777;font-size:11px}
  .grp{display:flex;align-items:center;gap:4px}
  #note{padding:2px 12px 4px;color:#999;font-size:11px;flex:0 0 auto}
</style></head>
<body><div id="page">
  <div id="head">@@HEADER@@</div>
  <div id="main">
    <div id="graphsec">
      <div class="row"><div class="rowhead"><select id="pick0"></select></div><div class="rowplot" id="row0"></div><div class="cursor" id="cur0"></div><div class="hline" id="hov0"></div></div>
      <div class="row"><div class="rowhead"><select id="pick1"></select></div><div class="rowplot" id="row1"></div><div class="cursor" id="cur1"></div><div class="hline" id="hov1"></div></div>
      <div class="row"><div class="rowhead"><select id="pick2"></select></div><div class="rowplot" id="row2"></div><div class="cursor" id="cur2"></div><div class="hline" id="hov2"></div></div>
    </div>
    <div id="scene3d"></div>
  </div>
  <div id="note"></div>
  <div id="bar">
    <div class="grp">
      <button id="first" title="first frame">|&lt;</button>
      <button id="prev" title="back one frame (left arrow)">&lt;</button>
      <button id="play" title="play / pause (space)">Play</button>
      <button id="next" title="forward one frame (right arrow)">&gt;</button>
      <button id="last" title="last frame">&gt;|</button>
    </div>
    <div class="grp"><label for="speed">speed</label>
      <select id="speed">
        <option value="1">1 frame / tick</option>
        <option value="2">2</option>
        <option value="5" selected>5</option>
        <option value="10">10</option>
        <option value="25">25</option>
      </select><span id="rate"></span>
    </div>
    <div class="grp"><label for="view">view</label>
      <select id="view"><option value="lab" selected>lab (whole flight)</option>
                        <option value="follow">follow the fly</option></select>
    </div>
    <div class="grp"><label for="units">x axis</label>
      <select id="units"><option value="frames" selected>frames</option>
                         <option value="ms">ms</option></select>
    </div>
    <input id="fader" type="range" min="0" value="0" step="1">
    <div id="read"></div>
  </div>
</div>
@@PLOTLY@@
<script>
"use strict";
const P = @@PAYLOAD@@;

/* ---- payload decoding -------------------------------------------------- */
/* Each array was embedded once as base64 float32; Plotly takes the resulting
   typed array directly, and NaN survives the round trip so unanalysed frames
   stay drawn as gaps. */
function unpack(o){
  const bin = atob(o.b), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}
const G = {};
for (const k in P.geom) G[k] = unpack(P.geom[k]);
const wsFull = {};
for (const k in P.ws) wsFull[k] = {phi: unpack(P.ws[k].phi), theta: unpack(P.ws[k].theta),
                                  psi: unpack(P.ws[k].psi)};

const N = P.n, NJ = 18, NROWS = @@NROWS@@, MAXTR = @@MAXTR@@;
const SCENE = "scene3d";
/* Trace indices in the 3D scene that move with time. Order must match the
   arrays pushed in sceneUpdate, and both mirror build_scene_figure. */
const DYN = [1,2,3,4,5,6,7,8,9,10,11,12];
const CONN = @@CONNECTIONS@@, VIS = @@VISIBLE@@;

let frame = 0, playing = false, follow = false, units = "frames", step = 5;
/* The visible index window. Zooming any time-series row narrows it, and every
   angle-space row is re-sliced to match -- that is the zoom link. */
let win = [0, N - 1];
let zooming = false;
const rowLabel = P.defaults.slice();
const rowKind = [null, null, null], rowSide = [null, null, null];
const sigByLabel = {};
P.signals.forEach(s => sigByLabel[s.label] = s);
const yCache = new Map();

const rowDiv = r => "row" + r;
/* Returns null rather than NaN for a non-numeric index, so a bad value
   stops here instead of reaching the readout as "frame NaN". */
const clampIdx = i => Number.isFinite(i)
  ? Math.max(0, Math.min(N - 1, Math.round(i))) : null;
function seriesY(sig, t){
  const key = sig.label + "|" + t;
  if (!yCache.has(key)) yCache.set(key, unpack(sig.traces[t].y));
  return yCache.get(key);
}
function toX(f){ return (units === "ms" && P.frame_rate) ? f / P.frame_rate * 1000 : f; }
function xvals(){ return units === "ms" && P.time_ms ? P.time_ms : P.frames; }
function xLabel(){ return units === "ms" ? P.x_label_ms : P.x_label_frames; }
function xToIndex(xv){
  const x = xvals(), d = x.length > 1 ? (x[1] - x[0]) : 1;
  return Math.round((xv - x[0]) / d);
}

/* ---- the 3D fly -------------------------------------------------------- */
function v3(a, i){ const o = 3*i; return [a[o], a[o+1], a[o+2]]; }
function jt(i, j){ const o = (i*NJ + j)*3; return [G.points[o], G.points[o+1], G.points[o+2]]; }
function arrow(originArr, vecArr, i, len){
  const o = v3(originArr, i), v = v3(vecArr, i);
  return [[o[0], o[0]+v[0]*len], [o[1], o[1]+v[1]*len], [o[2], o[2]+v[2]*len]];
}
function sceneUpdate(i){
  const xs = [], ys = [], zs = [];
  const lo = Math.max(0, i - P.trail);
  const tx = [], ty = [], tz = [];
  for (let k = lo; k <= i; k++){ const c = v3(G.cm, k); tx.push(c[0]); ty.push(c[1]); tz.push(c[2]); }
  xs.push(tx); ys.push(ty); zs.push(tz);
  /* skeleton: every bone in one NaN-separated polyline */
  const sx = [], sy = [], sz = [];
  for (const [a, b] of CONN){
    const pa = jt(i, a), pb = jt(i, b);
    sx.push(pa[0], pb[0], NaN); sy.push(pa[1], pb[1], NaN); sz.push(pa[2], pb[2], NaN);
  }
  xs.push(sx); ys.push(sy); zs.push(sz);
  const jx = [], jy = [], jz = [];
  for (const j of VIS){ const p = jt(i, j); jx.push(p[0]); jy.push(p[1]); jz.push(p[2]); }
  xs.push(jx); ys.push(jy); zs.push(jz);
  const bv = P.scales.body_vec, wv = P.scales.wing_vec;
  const arrows = [
    arrow(G.cm, G.x_body, i, bv), arrow(G.cm, G.y_body, i, bv), arrow(G.cm, G.z_body, i, bv),
    arrow(G.left_wing_CM,  G.left_wing_span,  i, wv), arrow(G.left_wing_CM,  G.left_wing_chord,  i, wv),
    arrow(G.right_wing_CM, G.right_wing_span, i, wv), arrow(G.right_wing_CM, G.right_wing_chord, i, wv),
  ];
  for (const a of arrows){ xs.push(a[0]); ys.push(a[1]); zs.push(a[2]); }
  const cx = [], cy = [], cz = [];
  for (let c = 0; c < 4; c++){ const o = (i*4 + c)*3;
    cx.push(G.stroke_corners[o]); cy.push(G.stroke_corners[o+1]); cz.push(G.stroke_corners[o+2]); }
  xs.push(cx); ys.push(cy); zs.push(cz);
  const c = v3(G.cm, i);
  xs.push([c[0]]); ys.push([c[1]]); zs.push([c[2]]);

  Plotly.restyle(SCENE, {x: xs, y: ys, z: zs}, DYN);
  if (follow){
    const h = P.scales.follow;
    Plotly.relayout(SCENE, {
      "scene.xaxis.range": [c[0]-h, c[0]+h],
      "scene.yaxis.range": [c[1]-h, c[1]+h],
      "scene.zaxis.range": [c[2]-h, c[2]+h]});
  }
}

/* ---- readout ----------------------------------------------------------- */
function pertLine(f){
  const p = P.pert;
  if (!p) return "";
  const rate = P.frame_rate, ms = d => rate ? (d / rate * 1000).toFixed(2) : String(d);
  if (f < p.onset) return "PRE  -" + ms(p.onset - f) + " ms";
  if (!p.end_known) return "PERT +" + ms(f - p.onset) + " ms  (end unrecorded)";
  if (f <= p.end)   return "PERT +" + ms(f - p.onset) + " ms";
  return "POST +" + ms(f - p.end) + " ms";
}
function readout(i){
  const f = P.frames[i];
  let s = "frame " + (f >= 0 ? "+" : "") + f;
  if (P.time_ms) s += "   " + (P.time_ms[i] >= 0 ? "+" : "") + P.time_ms[i].toFixed(2) + " ms";
  const p = pertLine(f);
  if (p) s += "\n" + p;
  document.getElementById("read").textContent = s;
}

/* ---- rows: either a time series or an angle-space scene ---------------- */
function wsSlice(r){
  const a = wsFull[rowSide[r]];
  Plotly.restyle(rowDiv(r), {
    x: [a.phi.subarray(win[0], win[1] + 1)],
    y: [a.theta.subarray(win[0], win[1] + 1)],
    z: [a.psi.subarray(win[0], win[1] + 1)],
    "marker.color": [P.frames.slice(win[0], win[1] + 1)]}, [0]);
}
/* Put a DOM overlay line at a data x on row r, or hide it. This is what the
   time cursor and the hover marker both use. Doing it in the DOM rather than
   as a plotly shape is what keeps playback smooth: a shape relayout redraws
   the panel's whole 4500-point trace, sixty times a second. */
function placeLine(el, r, xv){
  const gd = document.getElementById(rowDiv(r));
  const fl = gd._fullLayout;
  if (rowKind[r] !== "2d" || !fl || !fl.xaxis || !Number.isFinite(xv)){
    el.style.display = "none"; return;
  }
  const ax = fl.xaxis, ay = fl.yaxis;
  const px = ax._offset + (ax.d2p ? ax.d2p(xv) : ax.l2p(xv));
  if (px < ax._offset || px > ax._offset + ax._length){ el.style.display = "none"; return; }
  el.style.left = (gd.offsetLeft + px) + "px";
  el.style.top = (gd.offsetTop + ay._offset) + "px";
  el.style.height = ay._length + "px";
  el.style.display = "block";
}
function syncRow(r){
  if (rowKind[r] === "ws"){
    const a = wsFull[rowSide[r]];
    Plotly.restyle(rowDiv(r), {x: [[a.phi[frame]]], y: [[a.theta[frame]]],
                               z: [[a.psi[frame]]]}, [1]);
    document.getElementById("cur" + r).style.display = "none";
  } else if (rowKind[r] === "2d"){
    placeLine(document.getElementById("cur" + r), r, xvals()[frame]);
  }
}
/* The pointer's instant, shown on every OTHER panel. Deliberately does NOT
   touch `frame`: hovering a graph must not move the fly or the fader. */
let hoverIdx = null;
function setHover(i){
  hoverIdx = (i === null || !Number.isFinite(i)) ? null : clampIdx(i);
  for (let r = 0; r < NROWS; r++){
    const el = document.getElementById("hov" + r);
    if (rowKind[r] === "2d"){
      if (hoverIdx === null) el.style.display = "none";
      else placeLine(el, r, xvals()[hoverIdx]);
    } else if (rowKind[r] === "ws"){
      el.style.display = "none";
      const a = wsFull[rowSide[r]];
      Plotly.restyle(rowDiv(r), hoverIdx === null
        ? {x: [[]], y: [[]], z: [[]]}
        : {x: [[a.phi[hoverIdx]]], y: [[a.theta[hoverIdx]]], z: [[a.psi[hoverIdx]]]}, [2]);
    }
  }
}
/* Overlay positions depend on the axis-to-pixel mapping, so they have to be
   redone whenever that changes: a zoom, a resize, a panel swap. */
function refreshLines(){
  for (let r = 0; r < NROWS; r++) syncRow(r);
  if (hoverIdx !== null) setHover(hoverIdx);
}
function pertShapes(r){
  const p = P.pert;
  if (!p) return;
  const u = {};
  if (p.band){ u["shapes[0].x0"] = toX(p.band[0]); u["shapes[0].x1"] = toX(p.band[1]); }
  if (p.onset_visible){ u["shapes[1].x0"] = toX(p.onset); u["shapes[1].x1"] = toX(p.onset); }
  if (Object.keys(u).length) Plotly.relayout(rowDiv(r), u);
}
function setRow(r, label){
  const sig = sigByLabel[label];
  if (!sig) return;
  rowLabel[r] = label;
  const div = rowDiv(r);
  /* Only rebuild the plot when the KIND changes -- swapping one time series
     for another is a restyle, which keeps the user's zoom and is far cheaper. */
  if (rowKind[r] !== sig.kind || (sig.kind === "ws" && rowSide[r] !== sig.side)){
    const tpl = sig.kind === "ws" ? P.wsTemplate[sig.side] : P.rowTemplate;
    Plotly.newPlot(div, JSON.parse(JSON.stringify(tpl.data)),
                   JSON.parse(JSON.stringify(tpl.layout)),
                   {responsive: true, displaylogo: false});
    rowKind[r] = sig.kind; rowSide[r] = sig.side || null;
    attachRow(r);              // newPlot drops the div's event handlers
    if (sig.kind === "2d") pertShapes(r);
  }
  if (sig.kind === "ws"){
    wsSlice(r);
  } else {
    const x = xvals(), xs = [], ys = [], names = [], vis = [];
    for (let t = 0; t < MAXTR; t++){
      const has = t < sig.traces.length;
      xs.push(has ? x : []); ys.push(has ? seriesY(sig, t) : []);
      names.push(has ? sig.traces[t].name : ""); vis.push(has);
    }
    Plotly.restyle(div, {x: xs, y: ys, name: names, visible: vis},
                   [0, 1, 2].slice(0, MAXTR));
    const u = {"yaxis.title.text": sig.unit, "xaxis.title.text": xLabel()};
    if (win[0] > 0 || win[1] < N - 1){ u["xaxis.range"] = [x[win[0]], x[win[1]]]; }
    Plotly.relayout(div, u);
  }
  refreshLines();   // a swap rebuilds the panel, so the mapping is new
}

/* ---- the single entry point everything else calls ---------------------- */
function setFrame(i, moveFader){
  const j = clampIdx(i);
  if (j === null) return;
  frame = j;
  sceneUpdate(frame);
  for (let r = 0; r < NROWS; r++) syncRow(r);
  if (moveFader !== false) document.getElementById("fader").value = frame;
  readout(frame);
}

/* ---- zoom, linked across rows ------------------------------------------ */
/* Separate figures cannot use plotly's own xaxis.matches, so one row's zoom is
   pushed to the other time-series rows here and used to re-slice the
   angle-space rows. `zooming` stops the pushed relayouts echoing back. */
function setWindow(i0, i1, from){
  i0 = clampIdx(i0); i1 = clampIdx(i1);
  if (i0 === null || i1 === null || i1 <= i0) return;
  win = [i0, i1];
  zooming = true;
  const x = xvals();
  for (let r = 0; r < NROWS; r++){
    if (r === from) continue;
    if (rowKind[r] === "2d") Plotly.relayout(rowDiv(r), {"xaxis.range": [x[i0], x[i1]]});
    else if (rowKind[r] === "ws") wsSlice(r);
  }
  if (from !== undefined && rowKind[from] === "ws") wsSlice(from);
  zooming = false;
  refreshLines();
}
function onRelayout(r, ev){
  if (zooming) return;
  if ("xaxis.range[0]" in ev)
    setWindow(xToIndex(ev["xaxis.range[0]"]), xToIndex(ev["xaxis.range[1]"]), r);
  else if (ev["xaxis.autorange"] || ev["xaxis.range"] === undefined && ev.autosize)
    setWindow(0, N - 1, r);
  else refreshLines();   // a pan or a resize moves the mapping without changing the window
}
function attachRow(r){
  const gd = document.getElementById(rowDiv(r));
  /* Which frame a point on this row corresponds to. On an angle-space row the
     point index IS the frame index, offset by whatever window a zoom left
     visible; on a time series it comes from the x value. */
  const frameOf = pt => rowKind[r] === "ws" ? pt.pointNumber + win[0] : xToIndex(pt.x);
  /* Clicking seeks. Hovering does NOT -- it only marks the instant on the
     other panels, so running the pointer across a graph cannot drag the fly
     around with it. */
  gd.on("plotly_click", e => { pause(); setFrame(frameOf(e.points[0])); });
  gd.on("plotly_hover", e => setHover(frameOf(e.points[0])));
  gd.on("plotly_unhover", () => setHover(null));
  if (rowKind[r] === "2d") gd.on("plotly_relayout", ev => onRelayout(r, ev));
}

/* ---- playback ---------------------------------------------------------- */
function showRate(){
  /* What the chosen step actually means, since "slow motion" says nothing
     without a number: 60 ticks a second at the acquisition rate. */
  const el = document.getElementById("rate");
  el.textContent = P.frame_rate
    ? "= " + (step * 60 / P.frame_rate * 1000).toFixed(1) + " ms of flight per second (1/"
      + Math.round(P.frame_rate / (step * 60)) + " speed)"
    : "";
}
function tick(){
  if (!playing) return;
  /* Schedule the next frame BEFORE doing any work. If an update ever throws --
     a plotly call on a panel mid-swap, say -- playback drops that one frame
     instead of dying silently after the first. */
  requestAnimationFrame(tick);
  const n = frame + step;
  if (n >= N - 1){ setFrame(N - 1); pause(); return; }
  setFrame(n);
}
function play(){
  if (playing) return;                  // never let two loops run at once
  if (frame >= N - 1) setFrame(0);
  playing = true;
  document.getElementById("play").textContent = "Pause";
  requestAnimationFrame(tick);
}
function pause(){ playing = false; document.getElementById("play").textContent = "Play"; }

/* ---- wiring ------------------------------------------------------------ */
function buildPickers(){
  const groups = [];
  P.signals.forEach(s => { if (!groups.includes(s.group)) groups.push(s.group); });
  for (let r = 0; r < NROWS; r++){
    const sel = document.getElementById("pick" + r);
    for (const g of groups){
      const og = document.createElement("optgroup"); og.label = g;
      P.signals.filter(s => s.group === g).forEach(s => {
        const o = document.createElement("option");
        o.value = s.label; o.textContent = s.label;
        if (s.label === rowLabel[r]) o.selected = true;
        og.appendChild(o);
      });
      sel.appendChild(og);
    }
    sel.title = "signal shown in panel " + (r + 1);
    sel.addEventListener("change", e => setRow(r, e.target.value));
  }
}
function boot(){
  Plotly.newPlot(SCENE, P.scene.data, P.scene.layout, {responsive: true, displaylogo: false});
  buildPickers();
  for (let r = 0; r < NROWS; r++) setRow(r, rowLabel[r]);

  const fader = document.getElementById("fader");
  fader.max = N - 1;
  fader.addEventListener("input", e => { pause(); setFrame(+e.target.value, false); });

  document.getElementById("play").onclick  = () => playing ? pause() : play();
  document.getElementById("prev").onclick  = () => { pause(); setFrame(frame - 1); };
  document.getElementById("next").onclick  = () => { pause(); setFrame(frame + 1); };
  document.getElementById("first").onclick = () => { pause(); setFrame(0); };
  document.getElementById("last").onclick  = () => { pause(); setFrame(N - 1); };
  document.getElementById("speed").onchange = e => { step = +e.target.value; showRate(); };
  document.getElementById("view").onchange = e => {
    follow = e.target.value === "follow";
    /* The aspect ratio has to travel with the ranges to keep the axes equal:
       follow mode's box is a cube, so its ratio is 1:1:1, while the lab box is
       proportional to the flight's own extent. Neither is ever recomputed from
       the trace data -- see build_scene_figure for why that matters. */
    Plotly.relayout(SCENE, follow
      ? {"scene.aspectratio": {x: 1, y: 1, z: 1}}
      : {"scene.xaxis.range": [P.labRange[0][0], P.labRange[1][0]],
         "scene.yaxis.range": [P.labRange[0][1], P.labRange[1][1]],
         "scene.zaxis.range": [P.labRange[0][2], P.labRange[1][2]],
         "scene.aspectratio": {x: P.labAspect[0], y: P.labAspect[1],
                               z: P.labAspect[2]}});
    sceneUpdate(frame);
  };
  document.getElementById("units").onchange = e => {
    units = e.target.value;
    for (let r = 0; r < NROWS; r++){
      if (rowKind[r] !== "2d") continue;
      setRow(r, rowLabel[r]);
      pertShapes(r);
    }
    setFrame(frame);
  };
  document.addEventListener("keydown", e => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (e.key === " "){ e.preventDefault(); playing ? pause() : play(); }
    else if (e.key === "ArrowLeft"){ pause(); setFrame(frame - (e.shiftKey ? 10 : 1)); }
    else if (e.key === "ArrowRight"){ pause(); setFrame(frame + (e.shiftKey ? 10 : 1)); }
  });

  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(refreshLines, 120);   // after plotly's own reflow
  });

  document.getElementById("note").textContent = P.note;
  showRate();
  setFrame(0);
  // Axis offsets are only final once plotly has laid the panels out.
  setTimeout(refreshLines, 0);
}
boot();
</script></body></html>
"""


def render(h5_path, out_path, step, trail, cdn):
    with h5py.File(h5_path, "r") as h5:
        raw, geom, n, scales = build_geometry(h5, step)
        frames_all, rate, trigger_relative = frame_axis(h5, len(load(h5, "points_3D")))
        frames = np.asarray(frames_all, dtype=float)[::step][:n]
        signals, dropped = build_signals(h5, n, step, len(frames_all))
        if not signals:
            raise ValueError("no plottable signals in this h5")
        pert = build_perturbation(h5, frames)
        _, lbl_frames, _ = x_axis(frames, rate, "frames", trigger_relative)
        lbl_ms = x_axis(frames, rate, "ms", trigger_relative)[1] if rate else lbl_frames
        time_ms = (frames / rate * 1000).tolist() if rate else None

        stem = os.path.splitext(os.path.basename(h5_path))[0]
        header = " &middot; ".join(x for x in [
            f"<b>{stem}</b>",
            as_text(load(h5, "experiment")),
            f"{n} frames" + (f" of {len(frames_all)}" if step > 1 else ""),
            f"{rate:g} Hz" if rate else "",
            pert["label"] if pert else "",
        ] if x)

        scene_fig, lab_range, lab_aspect = build_scene_figure(raw, scales)
        # One empty template per row kind; the control layer newPlots a row
        # from whichever the chosen menu entry calls for.
        row_tpl = build_row_template(lbl_frames, pert)
        ws_tpl = {side: build_ws_template(side, lbl_frames)
                  for side in ("left", "right")}
        angles = {side: read_wing_angles(h5, side, n, step)
                  for side in ("left", "right")}

    note = ""
    if dropped:
        note = "not offered: " + "; ".join(dropped)
    if not trigger_relative:
        note = ("this movie predates the trigger datasets, so the x-axis is a box "
                "index and does NOT line up with the mp4 counter. " + note)

    payload = {
        "n": n, "trail": trail,
        "frames": [int(round(f)) for f in frames],
        "time_ms": [round(t, 4) for t in time_ms] if time_ms else None,
        "frame_rate": rate,
        "geom": geom, "scales": scales, "labRange": lab_range,
        "labAspect": lab_aspect,
        "signals": signals, "defaults": DEFAULT_ROWS,
        "pert": pert, "note": note,
        "x_label_frames": lbl_frames, "x_label_ms": lbl_ms,
        "ws": {side: {k: pack(v) for k, v in angles[side].items()}
               for side in angles},
        "scene": json.loads(scene_fig.to_json()),
        "rowTemplate": json.loads(row_tpl.to_json()),
        "wsTemplate": {side: json.loads(ws_tpl[side].to_json()) for side in ws_tpl},
    }

    if cdn:
        version = plotlyjs_version()
        script = f'<script src="{PLOTLY_CDN.format(version=version)}" charset="utf-8"></script>'
    else:
        script = "<script>" + get_plotlyjs() + "</script>"

    html = (HTML
            .replace("@@TITLE@@", stem)
            .replace("@@HEADER@@", header)
            .replace("@@CONNECTIONS@@", json.dumps(CONNECTIONS))
            .replace("@@VISIBLE@@", json.dumps(VISIBLE_JOINTS))
            .replace("@@NROWS@@", str(N_ROWS))
            .replace("@@MAXTR@@", str(MAX_TRACES_PER_ROW))
            .replace("@@PLOTLY@@", script)
            # Last, and with the payload as the replacement value rather than
            # part of the pattern, so a token-like string inside the data is
            # never rescanned.
            .replace("@@PAYLOAD@@", json.dumps(payload, allow_nan=False)))

    with open(out_path, "w") as f:
        f.write(html)
    return out_path


def plotlyjs_version():
    """The plotly.js the installed plotly ships, so --cdn serves a matching build."""
    import re
    m = re.search(r"plotly\.js v(\d+\.\d+\.\d+)", get_plotlyjs()[:2000])
    return m.group(1) if m else "3.1.1"


def make_viewer(h5_path, out=None, step=1, trail=400, cdn=False):
    """Write the viewer for one analysis h5. Returns the path, or None on failure.

    Named from the h5 stem rather than a constant, because collect_analysis_h5
    puts many movies' products side by side in one directory.
    """
    out = out or os.path.join(os.path.dirname(os.path.abspath(h5_path)),
                              os.path.splitext(os.path.basename(h5_path))[0] + OUT_SUFFIX)
    try:
        render(h5_path, out, step, trail, cdn)
    except Exception as e:
        print(f"flight viewer failed for {h5_path}: {e}", file=sys.stderr)
        return None
    print(f"  wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    return out


def find_h5(directory):
    return sorted(glob.glob(os.path.join(directory, PATTERN)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help=f"an *{SUFFIX} file, a directory holding one, "
                                   f"or a directory of such sub-directories")
    ap.add_argument("--out", default=None,
                    help="output path (single-file mode only)")
    ap.add_argument("-k", "--every", type=int, default=1, metavar="K",
                    help="keep every Kth frame. The default keeps all of them; "
                         "raise it only to shrink a file, since dropping frames "
                         "aliases the wingbeat (default: 1)")
    ap.add_argument("--trail", type=int, default=400, metavar="N",
                    help="length of the highlighted recent flight path, in "
                         "frames (default: 400)")
    ap.add_argument("--cdn", action="store_true",
                    help="load plotly.js from the CDN instead of inlining it: "
                         "~4.8 MB smaller per file, but needs a network "
                         "connection to open")
    args = ap.parse_args()

    if args.every < 1:
        sys.exit("--every must be at least 1")

    if os.path.isfile(args.target):
        sys.exit(0 if make_viewer(args.target, args.out, args.every, args.trail,
                                  args.cdn) else 1)

    if not os.path.isdir(args.target):
        sys.exit(f"Not a file or directory: {args.target}")

    direct = find_h5(args.target)
    if len(direct) > 1:
        sys.exit(f"Found {len(direct)} {PATTERN} files directly in {args.target}; "
                 f"expected exactly one for single mode.")
    if len(direct) == 1:
        sys.exit(0 if make_viewer(direct[0], args.out, args.every, args.trail,
                                  args.cdn) else 1)

    subdirs = sorted(d for d in (os.path.join(args.target, name)
                                 for name in os.listdir(args.target)) if os.path.isdir(d))
    n_ok = n_skip = 0
    for sub in subdirs:
        matches = find_h5(sub)
        if not matches:
            continue
        if len(matches) > 1:
            print(f"Skipping {sub}: found {len(matches)} {PATTERN} files (expected 1)",
                  file=sys.stderr)
            n_skip += 1
            continue
        if make_viewer(matches[0], None, args.every, args.trail, args.cdn):
            n_ok += 1
        else:
            n_skip += 1

    if n_ok == 0 and n_skip == 0:
        sys.exit(f"No {PATTERN} files found in {args.target} or its sub-directories.")
    print(f"done: {n_ok} written, {n_skip} skipped")


if __name__ == "__main__":
    main()
