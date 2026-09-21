"""Make every ensemble candidate agree on which wing is left, before the ensemble mixes them.

Predictor2D.harmonize_wing_labels lives here, apart from the rest of the predictor, because two
other programs need it without needing a pose estimator: code/realign_ensemble.py, which re-runs
the ensemble of an already-predicted movie, and code/local_reanalysis.py, which asks on a PC
whether a movie's ensemble would change at all. Importing Predictor pulls in torch, torchvision
and ultralytics; this module needs numpy alone.
"""
import warnings

import numpy as np

# how much better (metres) a candidate must fit the other candidates the other way round
WING_LABEL_SWAP_MIN_MARGIN = 0.001


def harmonize_wing_labels(all_points_list, min_margin=WING_LABEL_SWAP_MIN_MARGIN, n_iter=3):
    """Make every candidate agree on which wing is left before the ensemble mixes them.

    A candidate is one model's 3D reconstruction from one camera pair. Some models come out with
    the two wings' labels the other way round from the rest, for a whole movie or part of it, and
    the per-group selection below then took the median of a swapped candidate's 'left' wing (the
    fly's right one) together with the others' left wing -- landing both wings, and both hinges, on
    one physical wing. Frame by frame, each candidate's two wing centres are compared with the
    median over all candidates, and its left and right points (wing and hinge) are exchanged when
    it fits that median better the other way round by more than min_margin (metres). When nothing
    needs exchanging the input list itself is returned, so a movie whose models already agree is
    combined exactly as before. Returns (list, number of exchanged (frame, candidate) pairs).
    """
    num_joints = all_points_list[0].shape[1]
    points_per_wing = (num_joints - 2) // 2
    left = list(range(0, points_per_wing))
    right = list(range(points_per_wing, 2 * points_per_wing))
    wing_left, wing_right = left[:-1], right[:-1]      # the decision uses the wing points, not the hinge
    sizes = [points.shape[2] for points in all_points_list]
    stacked = np.concatenate(all_points_list, axis=2)   # (frames, joints, candidates, 3)
    swapped = np.zeros((stacked.shape[0], stacked.shape[2]), dtype=bool)
    work = None
    for _ in range(n_iter):
        source = stacked if work is None else work
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN candidates
            centre_left = np.nanmean(source[:, wing_left], axis=1)
            centre_right = np.nanmean(source[:, wing_right], axis=1)
            median_left = np.nanmedian(centre_left, axis=1, keepdims=True)
            median_right = np.nanmedian(centre_right, axis=1, keepdims=True)
        keep = (np.linalg.norm(centre_left - median_left, axis=-1)
                + np.linalg.norm(centre_right - median_right, axis=-1))
        exchange = (np.linalg.norm(centre_left - median_right, axis=-1)
                    + np.linalg.norm(centre_right - median_left, axis=-1))
        flip = (keep - exchange) > min_margin
        if not flip.any():
            break
        if work is None:
            work = stacked.copy()
        frames, candidates = np.nonzero(flip)
        chosen = work[frames, :, candidates]
        exchanged = chosen.copy()
        exchanged[:, left] = chosen[:, right]
        exchanged[:, right] = chosen[:, left]
        work[frames, :, candidates] = exchanged
        swapped ^= flip
    if work is None:
        return all_points_list, 0
    aligned, start = [], 0
    for size in sizes:
        aligned.append(work[:, :, start:start + size])
        start += size
    return aligned, int(swapped.sum())
