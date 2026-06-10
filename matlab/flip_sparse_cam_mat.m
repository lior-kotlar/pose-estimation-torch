function flip_sparse_cam_mat(mat_path)
% FLIP_SPARSE_CAM_MAT  Vertically flip a sparse-format camera .mat in place.
%
%   flip_sparse_cam_mat(MAT_PATH) loads the sparse movie at MAT_PATH (expected
%   to contain `metaData.bg`, `metaData.frameSize`, and a struct array `frames`
%   whose entries have `indIm` = [row col intensity] triplets), reflects every
%   frame's row coordinates (r -> frameSize(1)+1 - r), flips `metaData.bg`
%   vertically to match, and overwrites the original file under the same name.
%
%   Use this once on each cam5 sparse .mat from a mirror-camera lab setup so
%   that all downstream tooling (h5 dataset builder, easyWand-based calibration)
%   can treat all cameras uniformly.
%
%   Caveat: running this twice on the same file undoes the flip — there is no
%   built-in idempotency marker. Don't double-apply.
%
%   Example:
%       flip_sparse_cam_mat('/path/to/mov1/mov1_cam5_sparse.mat')

    if nargin < 1 || isempty(mat_path)
        error('flip_sparse_cam_mat:NoPath', 'Provide a path to a sparse .mat file.');
    end
    if ~isfile(mat_path)
        error('flip_sparse_cam_mat:FileNotFound', 'File not found: %s', mat_path);
    end

    S = load(mat_path);
    if ~isfield(S, 'metaData') || ~isfield(S, 'frames')
        error('flip_sparse_cam_mat:BadFormat', ...
              'Expected variables "metaData" and "frames" in %s', mat_path);
    end

    H = double(S.metaData.frameSize(1));
    S.metaData.bg = flipud(S.metaData.bg);

    n_frames = numel(S.frames);
    n_flipped = 0;
    for k = 1:n_frames
        if isempty(S.frames(k).indIm)
            continue
        end
        S.frames(k).indIm(:,1) = H + 1 - S.frames(k).indIm(:,1);
        n_flipped = n_flipped + 1;
    end

    save(mat_path, '-struct', 'S', '-v7.3');
    fprintf('Flipped %d/%d frames + bg (H=%d) in %s\n', ...
            n_flipped, n_frames, H, mat_path);
end
