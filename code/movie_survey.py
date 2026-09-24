"""What each movie still needs doing, so a run can do that and nothing else.

A predicted movie can be behind in three different ways, and they are not the same job:

    realign    one of the pose models labelled the two wings the other way round, so the ensemble
               mixed them into one physical wing. Only re-running the ensemble fixes it.
    reanalyse  the analysis code, the declaration or the 3D points have moved since the h5, CSV,
               plots and viewer were made.
    render     the overlay mp4 was made from points the movie no longer has.

Each is decided by a fingerprint rather than by a rule, which is what makes the dependency between
them trustworthy: installing a new ensemble changes the points fingerprint, which makes the
analysis stale, which changes the analysed points, which makes the video stale. Nothing has to
remember to cascade -- it falls out. So a run surveys, does one stage, and surveys again.

The survey costs almost nothing except the realign question, which has to read every model's
candidates (about 18 MB a movie). That answer is cached beside the movie and keyed on those files,
so only a movie whose models actually changed pays it twice.
"""
import json
import os

STAGES = ('realign', 'reanalyse', 'render')
# the screening answer, cached beside the movie it is about
SCREEN_CACHE = '.ensemble_screen.json'
POINTS_ALL = 'points_3D_all.npy'
REALIGN_MARKER = '.realigned_ensemble.json'
BLOCKED_MARKER = '.realign_blocked.json'


def ensemble_members(movie_dir):
    """The member folders of a movie that hold 3D candidates, in the ensemble's own order.

    Archived and hidden folders are passed over, so an earlier realignment's superseded_ensemble_*
    or .realign_staging can never be taken for a member."""
    try:
        names = os.listdir(movie_dir)
    except OSError:
        return []
    return sorted(os.path.join(movie_dir, name) for name in names
                  if not (name.startswith('superseded_') or name.startswith('.'))
                  and os.path.isfile(os.path.join(movie_dir, name, POINTS_ALL)))


def member_signature(members):
    """What identifies this movie's candidates without reading them: name, size and time."""
    signature = []
    for member in members:
        path = os.path.join(member, POINTS_ALL)
        try:
            stat = os.stat(path)
        except OSError:
            return None
        signature.append([os.path.basename(member), stat.st_size, stat.st_mtime_ns])
    return signature


def screen_movie(movie_dir, force=False):
    """How many (frame, candidate) pairs would be exchanged if this ensemble were re-run.

    Zero means the ensemble would come out identical and there is nothing to realign. The answer
    is cached beside the movie against the candidate files it was computed from, because reading
    them is the one genuinely expensive part of a survey."""
    members = ensemble_members(movie_dir)
    if len(members) < 2:
        return 0
    signature = member_signature(members)
    cache_path = os.path.join(movie_dir, SCREEN_CACHE)
    if not force and signature is not None:
        try:
            with open(cache_path, encoding='utf-8') as f:
                cached = json.load(f)
            if cached.get('members') == signature:
                return int(cached.get('exchanged_pairs', 0))
        except (OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError):
            pass

    from wing_labels import harmonize_wing_labels
    import numpy as np
    points = [np.load(os.path.join(member, POINTS_ALL)) for member in members]
    _, exchanged = harmonize_wing_labels(points)
    exchanged = int(exchanged)
    if signature is not None:
        try:
            staged = cache_path + '.partial'
            with open(staged, 'w', encoding='utf-8') as f:
                json.dump({'members': signature, 'exchanged_pairs': exchanged}, f)
            os.replace(staged, cache_path)
        except OSError:
            pass            # a read-only disk costs speed, never correctness
    return exchanged


def survey(movie_dirs, code_fp, path_maps=(), dataset_roots=(), group_of=None,
           retry_blocked=False, render_unknown=False, show=False, on_progress=None):
    """One row per movie saying what it needs. Reads only this machine; writes only the cache."""
    import reanalyse_movies as rm

    rows = rm.preflight_rows(movie_dirs, 'auto', path_maps, code_fp, group_of=group_of,
                             dataset_roots=dataset_roots, show=show)
    for number, row in enumerate(rows, 1):
        movie_dir = row['movie_dir']
        if on_progress:
            on_progress(number, len(rows))
        realigned = os.path.isfile(os.path.join(movie_dir, REALIGN_MARKER))
        blocked = os.path.isfile(os.path.join(movie_dir, BLOCKED_MARKER))
        exchanged = 0 if (realigned or (blocked and not retry_blocked)) else screen_movie(movie_dir)
        recorded = rm.recorded_box_path(movie_dir, None)
        source = rm.reachable(recorded, path_maps, dataset_roots) if recorded else ''
        has_source = bool(source) and os.path.isfile(source)
        video = rm.video_state(movie_dir)
        row.update(exchanged_pairs=exchanged, realigned=realigned, blocked=blocked,
                   video=video, source=source if has_source else '',
                   recorded_source=recorded or '')
        row['needs'] = movie_needs(row, render_unknown=render_unknown)
    return rows


def movie_needs(row, render_unknown=False):
    """The stages a movie still needs, in the order they have to happen.

    A movie that needs realigning is counted for all three, because a successful realignment makes
    its analysis and its video stale by construction. That is an estimate, not a promise: the
    cluster refuses a realignment that would make anything worse, and the run surveys again after
    every stage, so a refused movie is quietly dropped from the later ones rather than redone for
    nothing."""
    needs = []
    if row.get('exchanged_pairs'):
        needs.append('realign')
    if row.get('state') != 'current' or 'realign' in needs:
        needs.append('reanalyse')
    renderable = bool(row.get('source'))
    wants_render = row.get('video') in ('stale', 'missing') or 'realign' in needs
    if row.get('video') == 'unknown' and render_unknown:
        wants_render = True
    if wants_render and renderable:
        needs.append('render')
    return tuple(needs)


def tally(rows):
    """How many movies need each stage, and how many need nothing at all."""
    counts = {stage: sum(1 for row in rows if stage in row.get('needs', ())) for stage in STAGES}
    counts['nothing'] = sum(1 for row in rows if not row.get('needs'))
    return counts


def summarise(rows, show_rows=12):
    """The plan, as a table per experiment and a line of totals. Returns the lines."""
    groups = sorted({row['group'] for row in rows})
    width = max([len(g) for g in groups] + [10])
    lines = [f"{'experiment'.ljust(width)}  movies  realign  re-analyse  render  nothing"]
    for group in groups[:show_rows]:
        subset = [row for row in rows if row['group'] == group]
        counts = tally(subset)
        lines.append(f"{group.ljust(width)}  {len(subset):6d}  {counts['realign']:7d}  "
                     f"{counts['reanalyse']:10d}  {counts['render']:6d}  {counts['nothing']:7d}")
    if len(groups) > show_rows:
        lines.append(f"... and {len(groups) - show_rows} more experiment(s)")

    counts = tally(rows)
    todo = ', '.join(f"{stage} {counts[stage]}" for stage in STAGES if counts[stage])
    lines.append('')
    lines.append(f"to do   : {todo or 'nothing'}")
    if counts['nothing']:
        lines.append(f"skipping: {counts['nothing']} movie(s) already up to date")
    unknown = [row for row in rows if row.get('video') == 'unknown']
    if unknown:
        lines.append(f"unsure about the video of {len(unknown)} movie(s), made before videos were "
                     f"stamped; --render-unknown redoes them")
    missing = [row for row in rows if not row.get('source')]
    if missing:
        lines.append(f"cannot render {len(missing)} movie(s): their source box h5 was not found "
                     f"(--dataset-root says where to look)")
    return lines


def with_stage(rows, stage):
    """The movie directories that still need one stage."""
    return [row['movie_dir'] for row in rows if stage in row.get('needs', ())]
