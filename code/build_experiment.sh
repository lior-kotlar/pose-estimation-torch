#!/usr/bin/env bash
# build_experiment.sh
#
# Build h5 movie files (one per movie) and one calibration.h5 (per experiment)
# by driving the MATLAB scripts in matlab/.
#
# Usage:
#   build_experiment.sh <input_dir> <easywand.mat> [--max-frames N] [--start N] [--end N]
#
# Modes (auto-detected from <input_dir>):
#
#   1. Single-movie mode: <input_dir> directly contains 4 *_sparse.mat files
#      named ..._cam<N>_sparse.mat. The directory's basename must be "mov<N>"
#      (e.g. mov1, mov3). One h5 is written into <input_dir>.
#
#   2. Multi-movie mode: <input_dir> contains many mov<N>/ subdirectories, each
#      with 4 *_sparse.mat files. One h5 is written into each movie subdir.
#
# After all h5 files are written, a single calibration.h5 is created at
# <input_dir>/calibration.h5 from <easywand.mat>.

set -euo pipefail

# ----- paths ---------------------------------------------------------------
REPO_ROOT="/cs/labs/tsevi/lior.kotlar/pose-estimation-torch"
DATASET_SCRIPT="$REPO_ROOT/matlab/CreateDatasetHDF5_from_list_fixed.m"
CALIB_SCRIPT="$REPO_ROOT/matlab/run after experiment/create_camera_calibration_h5.m"
MATLAB_BIN="${MATLAB_BIN:-matlab}"

# ----- args ---------------------------------------------------------------
usage() {
    sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}

[ $# -ge 2 ] || usage
INPUT_DIR="$(realpath "$1")"
EASYWAND_MAT="$(realpath "$2")"
shift 2

MAX_FRAMES=""
START_IND=""
END_IND=""
while [ $# -gt 0 ]; do
    case "$1" in
        --max-frames) MAX_FRAMES="$2"; shift 2 ;;
        --start)      START_IND="$2";  shift 2 ;;
        --end)        END_IND="$2";    shift 2 ;;
        -h|--help)    usage ;;
        *) echo "Unknown arg: $1"; usage ;;
    esac
done

[ -d "$INPUT_DIR" ] || { echo "Input dir not found: $INPUT_DIR" >&2; exit 1; }
[ -f "$EASYWAND_MAT" ] || { echo "easyWand .mat not found: $EASYWAND_MAT" >&2; exit 1; }

# ----- helpers ------------------------------------------------------------
# Count *_sparse.mat files directly in a dir (no recursion).
count_sparse_mats() {
    find "$1" -maxdepth 1 -mindepth 1 -name '*sparse.mat' -type f | wc -l
}

# Parse the integer N from a "movN" basename, case-insensitively and ignoring
# leading zeros (mov3, Mov003, MOV03 all -> 3). Echoes N, or empty on no match.
parse_movie_num() {
    local base
    base="$(basename "$1")"
    local n=""
    shopt -s nocasematch
    [[ "$base" =~ ^mov([0-9]+)$ ]] && n="$((10#${BASH_REMATCH[1]}))"
    shopt -u nocasematch
    echo "$n"
}

# MATLAB's dataset script hard-codes the movie subdir as ['mov', int2str(movie_num)]
# -- lowercase 'mov', no leading zeros. Raw Windows-exported dirs look like 'Mov001',
# so rename <dir> to the canonical <parent>/mov<N> that MATLAB expects. Echoes the
# canonical path on stdout; returns nonzero (with a message) on a name collision.
canonical_movie_dir() {
    local dir="$1" mn="$2"
    local canon
    canon="$(dirname "$dir")/mov$mn"
    if [ "$dir" != "$canon" ]; then
        if [ -e "$canon" ]; then
            echo "  Cannot rename $(basename "$dir") -> mov$mn (target already exists)" >&2
            return 1
        fi
        mv "$dir" "$canon"
        echo "  Renamed $(basename "$dir") -> mov$mn (pipeline convention)" >&2
    fi
    echo "$canon"
}

# Build the MATLAB -batch prelude with all overrides for the dataset script.
# Args: <sparse_folder_path> <save_path> <movie_num>
matlab_dataset_cmd() {
    local sfp="$1" sp="$2" mn="$3"
    local cmd="sparse_folder_path='$sfp'; save_path='$sp'; movie_num=$mn;"
    [ -n "$START_IND"  ] && cmd+=" start_ind=$START_IND;"
    [ -n "$END_IND"    ] && cmd+=" end_ind=$END_IND;"
    [ -n "$MAX_FRAMES" ] && cmd+=" max_frames=$MAX_FRAMES;"
    cmd+=" run('$DATASET_SCRIPT')"
    echo "$cmd"
}

# ----- detect mode & enumerate jobs ---------------------------------------
declare -a JOBS_SFP  # sparse_folder_path per job
declare -a JOBS_SP   # save_path per job
declare -a JOBS_MN   # movie_num per job
njobs=0              # scalar count (nounset-safe on older bash w/ empty arrays)

n_direct=$(count_sparse_mats "$INPUT_DIR")

if [ "$n_direct" -eq 4 ]; then
    # Single-movie mode
    mn=$(parse_movie_num "$INPUT_DIR")
    if [ -z "$mn" ]; then
        echo "Single-movie mode: input dir basename must be 'mov<N>' (got '$(basename "$INPUT_DIR")')" >&2
        echo "Rename it or move the 4 mats into a mov<N>/ subdirectory." >&2
        exit 1
    fi
    INPUT_DIR=$(canonical_movie_dir "$INPUT_DIR" "$mn") || exit 1
    JOBS_SFP+=("$(dirname "$INPUT_DIR")")
    JOBS_SP+=("$INPUT_DIR")
    JOBS_MN+=("$mn")
    njobs=$((njobs + 1))
    echo "Mode: single-movie. Movie #$mn at $INPUT_DIR"
elif [ "$n_direct" -eq 0 ]; then
    # Multi-movie mode: scan for movN/ subdirs with 4 mats
    while IFS= read -r -d '' sub; do
        mn=$(parse_movie_num "$sub")
        [ -z "$mn" ] && continue
        n=$(count_sparse_mats "$sub")
        if [ "$n" -eq 4 ]; then
            canon=$(canonical_movie_dir "$sub" "$mn") || continue
            JOBS_SFP+=("$INPUT_DIR")
            JOBS_SP+=("$canon")
            JOBS_MN+=("$mn")
            njobs=$((njobs + 1))
        else
            echo "Skipping $sub (expected 4 *_sparse.mat, found $n)" >&2
        fi
    done < <(find "$INPUT_DIR" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)
    if [ "$njobs" -eq 0 ]; then
        echo "Multi-movie mode: no mov<N>/ subdirs with 4 *_sparse.mat found in $INPUT_DIR" >&2
        exit 1
    fi
    echo "Mode: multi-movie. $njobs movie(s) queued: ${JOBS_MN[*]}"
else
    echo "Ambiguous: $INPUT_DIR contains $n_direct *_sparse.mat (expected 0 or 4)" >&2
    exit 1
fi

# ----- run dataset jobs ---------------------------------------------------
for i in "${!JOBS_MN[@]}"; do
    sfp="${JOBS_SFP[$i]}"
    sp="${JOBS_SP[$i]}"
    mn="${JOBS_MN[$i]}"
    echo
    echo "===== Building h5 for mov$mn ====="
    echo "  sparse_folder_path = $sfp"
    echo "  save_path          = $sp"
    cmd=$(matlab_dataset_cmd "$sfp" "$sp" "$mn")
    echo "  MATLAB: $cmd"
    "$MATLAB_BIN" -batch "$cmd"
done

# ----- create calibration -------------------------------------------------
CALIB_OUT="$INPUT_DIR/calibration.h5"
echo
echo "===== Building calibration.h5 ====="
echo "  easy_wand_path = $EASYWAND_MAT"
echo "  savePath       = $CALIB_OUT"

if [ -f "$CALIB_OUT" ]; then
    echo "  (existing $CALIB_OUT will be deleted to avoid h5create collision)"
    rm -f "$CALIB_OUT"
fi

calib_cmd="easy_wand_path='$EASYWAND_MAT'; savePath='$CALIB_OUT'; run('$CALIB_SCRIPT')"
echo "  MATLAB: $calib_cmd"
"$MATLAB_BIN" -batch "$calib_cmd"

echo
echo "Done. Outputs:"
for i in "${!JOBS_MN[@]}"; do
    echo "  ${JOBS_SP[$i]}/mov_${JOBS_MN[$i]}_*_ds_3tc_7tj.h5"
done
echo "  $CALIB_OUT"
