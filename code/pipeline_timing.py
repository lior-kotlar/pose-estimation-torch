"""Per-step, per-movie timing ledger for the prep+predict pipeline.

Writers may live in different processes (the CPU prep job + N predict array
tasks on different GPU nodes), so the ledger is an append-only CSV guarded
by an advisory file lock. Each step appends one row; you reconstruct
end-to-end per-movie time with `groupby(movie)` in pandas (or
`max(ended_at) - min(started_at)` for wall-clock including idle time).
"""
import fcntl
import os

CSV_HEADER = "movie,step,started_at,ended_at,elapsed_s,n_frames\n"


def record(csv_path, movie, step, started_at, ended_at, n_frames=None):
    """Append one (movie, step, started_at, ended_at, elapsed, n_frames) row.

    `n_frames` is the post-intersection frame count for this movie when the
    caller knows it (build, verify, predict). Steps that don't know it
    (flip, plot) pass None and the field is written empty.

    Writes the header if the file is new. Safe under concurrent writers
    via fcntl.flock. csv_path=None or '' is a no-op so callers can pass
    through an unset config without a try/except.
    """
    if not csv_path:
        return
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    n_str = "" if n_frames is None else str(int(n_frames))
    line = (f"{movie},{step},{started_at:.3f},{ended_at:.3f},"
            f"{ended_at - started_at:.3f},{n_str}\n")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            if new:
                f.write(CSV_HEADER)
            f.write(line)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def earliest_start(csv_path, movie):
    """Earliest started_at recorded for `movie`, or None if the ledger has
    no rows for that movie yet. Used by predict.py to write a "total" row
    spanning from the prep job's first step through the end of plot."""
    if not csv_path or not os.path.exists(csv_path):
        return None
    earliest = None
    with open(csv_path) as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            next(f, None)  # skip header
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) >= 3 and parts[0] == movie:
                    try:
                        t = float(parts[2])
                    except ValueError:
                        continue
                    if earliest is None or t < earliest:
                        earliest = t
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return earliest


def latest_n_frames(csv_path, movie):
    """Most-recently-recorded non-empty n_frames for `movie`, or None.
    Lets the "total" row inherit the count from earlier steps without the
    caller having to thread it through every code path."""
    if not csv_path or not os.path.exists(csv_path):
        return None
    latest = None
    with open(csv_path) as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            next(f, None)  # skip header
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) >= 6 and parts[0] == movie and parts[5]:
                    try:
                        latest = int(parts[5])
                    except ValueError:
                        continue
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return latest
