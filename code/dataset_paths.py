"""Find a movie's source dataset wherever it lives now, and shrink one for transport.

A predicted movie records where its source box h5 and calibration were at predict time, in the
member configuration.json the prediction wrote. Those are the cluster's paths, in two forms that
both exist in the wild:

    /cs/labs/tsevi/lior.kotlar/pose-estimation-torch/inference_datasets/Tsory/ex.../mov11/mov_11_....h5
    inference_datasets/roni_dark/2023_08_07_10ms/1to30/mov5/mov_5_....h5   (relative to the project)

The datasets themselves are far too large to keep on the cluster, so those paths go stale as soon
as an experiment is moved off it -- and for the movies predicted before source.json carried
provenance, that recorded path is the only record of where the data came from. resolve() turns
either form into wherever the file lives on this machine, given the roots its owner keeps datasets
under.

Everything a movie's analysis and video need sits next to the box h5 -- the calibration one level
up, prescan_cam_validity.npz and the sparse mats beside it, perturbation.json in the experiment --
and every one of those is found by dirname() of the box path (utils.resolve_calibration_path,
load_cam_validity, get_trigger_frame_info, resolve_perturbation_path). So resolving the box
resolves all of them.

A movie predicted from a PC (local_predict.py) comes home with only a render-only copy of its box,
`<stem>_render.h5` beside where the full box would be: resolve() falls back to it, so the movie's
video can still be rebuilt, while predict skips such files (they hold no images for most channels).

The path functions are standard library only, so anything can import them cheaply; reduce_box
needs h5py and numpy and imports them when it is called.

    python code/dataset_paths.py reduce SRC_BOX DST_BOX
    python code/dataset_paths.py reduce-movie MOVIE_DIR OUT_DIR   # -> OUT_DIR/<stem>_render.h5
"""
import glob
import json
import os
import sys

# the folder that holds every experiment's source data, and the segment both recorded forms share
DATASETS_SEGMENT = 'inference_datasets'
# how many trailing components a root may be given as, when it names an experiment rather than the
# datasets folder: <experiment>/<movN>/<file>
TAIL_DEPTH = 3


def split_path(recorded):
    """A recorded path's components, whichever separator the machine that wrote it used."""
    return [part for part in recorded.replace('\\', '/').split('/') if part not in ('', '.')]


def index_key(path):
    """What a file is indexed under: its name, lowercased.

    The name alone does not identify a movie -- mov_11_... occurs in many experiments -- so it is
    only ever a shortlist, narrowed by best_match below."""
    return split_path(path)[-1].lower()


def shared_tail(a, b):
    """How many trailing components two paths have in common, ignoring case."""
    first, second = [p.lower() for p in split_path(a)], [p.lower() for p in split_path(b)]
    shared = 0
    for one, other in zip(reversed(first), reversed(second)):
        if one != other:
            break
        shared += 1
    return shared


def best_match(recorded, candidates):
    """The candidate that agrees with the recorded path for longest, when just one does.

    A local layout may differ from the cluster's anywhere above the file, so the match is scored
    rather than required. A tie resolves to nothing: two movies of the same name in different
    experiments must not be told apart by a guess."""
    if not candidates:
        return None
    scored = [(shared_tail(recorded, path), path) for path in candidates]
    best = max(score for score, _ in scored)
    winners = [path for score, path in scored if score == best]
    return winners[0] if len(winners) == 1 else None


def below_datasets(recorded, segment=DATASETS_SEGMENT):
    """The part of a recorded path below its inference_datasets segment, or None.

    Both recorded forms carry that segment, so taking what follows it turns an absolute cluster
    path and a project-relative one into the same thing: a path relative to whatever folder holds
    the experiments on this machine."""
    parts = split_path(recorded)
    lowered = [part.lower() for part in parts]
    if segment.lower() not in lowered:
        return None
    # the last occurrence, so a root that itself sits under a folder of that name still works
    after = len(lowered) - 1 - lowered[::-1].index(segment.lower())
    rest = parts[after + 1:]
    return os.path.join(*rest) if rest else None


# the name a render-only box goes by, beside where its full box would be
RENDER_SUFFIX = '_render.h5'


def render_box_name(box_path):
    """mov_2_448_2330_ds_3tc_7tj.h5 -> mov_2_448_2330_ds_3tc_7tj_render.h5"""
    return box_path[:-len('.h5')] + RENDER_SUFFIX


def is_render_box(path):
    return path.lower().endswith(RENDER_SUFFIX)


def _existing(path):
    """`path` when it is a file; for a box h5 that is not, its render-only sibling when that is."""
    if os.path.isfile(path):
        return path
    if path.lower().endswith('.h5') and not is_render_box(path):
        sibling = render_box_name(path)
        if os.path.isfile(sibling):
            return sibling
    return None


def resolve(recorded, roots=(), index=None):
    """Where a recorded source path is on this machine, or None.

    Tried in order: the path as recorded, which is what keeps the cluster working unchanged; the
    part below inference_datasets joined to each root, which covers both recorded forms in one
    rule; then the index, for a local layout that does not mirror the cluster's. An ambiguous
    index match resolves to nothing rather than to a guess. At every step a box h5 that is not
    there is also looked for as its render-only copy (RENDER_SUFFIX), after the box itself."""
    if not recorded:
        return None
    found = _existing(recorded)
    if found:
        return found
    rest = below_datasets(recorded)
    for root in roots:
        if rest:
            found = _existing(os.path.join(root, rest))
            if found:
                return found
        # a root given as the experiment folder itself, rather than the datasets folder
        found = _existing(os.path.join(root, *split_path(recorded)[-TAIL_DEPTH:]))
        if found:
            return found
    if index:
        names = [index_key(recorded)]
        if names[0].endswith('.h5') and not is_render_box(names[0]):
            names.append(index_key(render_box_name(recorded)))
        for name in names:
            found = best_match(recorded, index.get(name) or [])
            if found and os.path.isfile(found):
                return found
    return None


def build_index(roots, suffixes=('.h5', '.npz', '.json', '.mat')):
    """Map every interesting file under the roots to its name.

    Only needed when a local layout does not mirror the cluster's -- a root whose experiments sit
    where the recorded paths say they do never reaches this. Walking an external disk is slow, so
    the caller caches the result and rebuilds it only when a lookup misses."""
    index = {}
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames
                                 if not (d.startswith('.') or d.startswith('superseded_')))
            for name in filenames:
                if name.lower().endswith(suffixes):
                    path = os.path.join(dirpath, name)
                    index.setdefault(index_key(path), []).append(path)
    return index


def load_index(cache_path):
    try:
        with open(cache_path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_index(cache_path, index):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    staged = cache_path + '.partial'
    with open(staged, 'w', encoding='utf-8') as f:
        json.dump(index, f)
    os.replace(staged, cache_path)


class Resolver:
    """resolve() with the index built once, lazily, and only if the roots need it."""

    def __init__(self, roots=(), cache_path=None):
        self.roots = [os.path.abspath(r) for r in roots if r]
        self.cache_path = cache_path
        self._index = None
        self._tried_index = False

    def index(self):
        if self._index is None and not self._tried_index:
            self._tried_index = True
            self._index = (load_index(self.cache_path) if self.cache_path else None)
            if self._index is None and self.roots:
                self._index = build_index(self.roots)
                if self.cache_path:
                    save_index(self.cache_path, self._index)
        return self._index

    def __call__(self, recorded):
        if not recorded:
            return None
        direct = resolve(recorded, self.roots)
        if direct or not self.roots:
            return direct
        return resolve(recorded, (), self.index())


# -- shrinking a box for transport ----------------------------------------------------------------

# Visualizer.create_movie_mp4 draws one time-channel per camera: the box's channel axis is
# num_time_channels per camera, cameras in order, and it reads channel 1 of each.
TIME_CHANNELS = 3
BOX = 'box'
REDUCED_ATTR = 'reduced_channels'


def render_channels(num_channels, time_channels=TIME_CHANNELS):
    """The channels the mp4 renderer reads: [1, 4, 7] for 3 cameras, [1, 4, 7, 10] for 4."""
    return [time_channels * cam + 1 for cam in range(num_channels // time_channels)]


def reduce_box(src_path, dst_path, frames_per_block=64):
    """Copy a box h5, keeping only the image channels the mp4 renderer reads.

    The box is nine channels of 192x192 floats per frame -- 2.5 GB for a long movie, gzipped to
    about 185 MB -- and the renderer reads three of them. Zeroing the rest leaves a file about a
    third the size that gzip squashes to nothing, which is what makes rendering on the cluster
    affordable for datasets that only exist on a PC.

    The shape is kept exactly, so the renderer's own `n_cams = shape[1] // 3` still holds and no
    rendering code has to know about this. The kept channels are recorded on the dataset, and
    render-side code is expected to refuse a box whose recorded channels are not the ones it
    reads -- so a future change to the renderer fails loudly instead of drawing black panels.

    Returns (kept channels, source bytes, destination bytes).
    """
    import h5py
    import numpy as np

    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        box = src[BOX]
        keep = render_channels(box.shape[1])
        out = dst.create_dataset(BOX, shape=box.shape, dtype=box.dtype,
                                 chunks=box.chunks, compression='gzip')
        out.attrs[REDUCED_ATTR] = np.asarray(keep, dtype='int64')
        for start in range(0, box.shape[0], frames_per_block):
            stop = min(start + frames_per_block, box.shape[0])
            block = np.zeros((stop - start,) + box.shape[1:], dtype=box.dtype)
            block[:, keep] = box[start:stop, keep]
            out[start:stop] = block
        # everything else in a box is small bookkeeping (cropzone, frameInds); the cropzone is
        # what the reprojection needs, so copy them all rather than guess
        for name, item in src.items():
            if name != BOX and isinstance(item, h5py.Dataset):
                dst.create_dataset(name, data=item[()], compression='gzip')
    return keep, os.path.getsize(src_path), os.path.getsize(dst_path)


def reduced_channels(box_path):
    """Which channels a box still holds images for, or None when it is an unreduced one."""
    import h5py
    with h5py.File(box_path, 'r') as f:
        value = f[BOX].attrs.get(REDUCED_ATTR)
    return None if value is None else [int(v) for v in value]


def movie_box(movie_dir):
    """The full box h5 prep built in a movie folder, or None (render-only copies never count)."""
    found = sorted(path for path in glob.glob(os.path.join(movie_dir, 'mov_*_ds_*tc_*tj.h5'))
                   if not is_render_box(path))
    return found[0] if found else None


def reduce_movie(movie_dir, out_dir):
    """Write <out_dir>/<stem>_render.h5 from the movie's box; returns its path. Lands under a
    temporary name first, so a half-written copy never passes for a finished one."""
    box = movie_box(movie_dir)
    if box is None:
        raise FileNotFoundError(f'no box h5 in {movie_dir}')
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, os.path.basename(render_box_name(box)))
    staged = target + '.partial'
    keep, before, after = reduce_box(box, staged)
    os.replace(staged, target)
    print(f'render box {os.path.basename(target)}: channels {keep}, '
          f'{before / 1e6:.0f} -> {after / 1e6:.0f} MB', flush=True)
    return target


def main(argv):
    if len(argv) == 3 and argv[0] == 'reduce':
        keep, before, after = reduce_box(argv[1], argv[2])
        print(f'channels {keep}: {before / 1e6:.0f} -> {after / 1e6:.0f} MB')
        return 0
    if len(argv) == 3 and argv[0] == 'reduce-movie':
        reduce_movie(argv[1], argv[2])
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
