function resave_sparse_v73(list_path)
% RESAVE_SPARSE_V73  Re-save old-format sparse cam .mat files as v7.3 (HDF5).
%
%   resave_sparse_v73(LIST_PATH) reads a text file with one
%   "<src_path>\t<dst_path>" pair per line and, for each, loads the sparse
%   movie at src and writes it to dst with '-v7.3'.
%
%   WHY THIS EXISTS
%   ---------------
%   Older mirror-cam flip scripts end with a bare `save(path,...)`. Without
%   '-v7.3' that downgrades the file to the MATLAB 5.0 / v7 container. Any
%   experiment flipped that way arrives with its mirror cam unreadable by
%   h5py (e.g. every cam1 mat of the "dark 24-1" archive, all of them
%   carrying metaData.isFlipped == 1).
%
%   Three parts of the pipeline can't read it:
%     * code/scan_sparse_movies.py (the prescan) opens mats with h5py and
%       hard-fails -> every movie is reported ERR and dropped;
%     * CreateDatasetHDF5_from_list_fixed.m uses matfile() with partial
%       indexing (x.frames(a:b,1)), which MATLAB refuses on non-v7.3 files;
%     * utils.get_trigger_frame_info reads metaData.startFrame with h5py
%       from the FIRST sparse mat alphabetically (= cam1), so it silently
%       returns (None, None) and the analysis h5 loses trigger-relative
%       frame numbering.
%
%   THIS DOES NOT FLIP ANYTHING. The source mats are already flipped; the
%   downgrade is a side effect of that flip, not something to re-apply. The
%   isFlipped guard below refuses any mat that is not in that state, so an
%   unflipped archive fails loudly rather than converting silently.
%   Running the pipeline over the output requires --skip-flip.
%
%   Writes to a NEW path, never in place, so the original stays untouched.
%
%   Example:
%       matlab -batch "resave_sparse_v73('/tmp/convert_list.tsv')"

    if nargin < 1 || isempty(list_path)
        error('resave_sparse_v73:NoList', 'Provide a path to the TSV list.');
    end
    if ~isfile(list_path)
        error('resave_sparse_v73:ListNotFound', 'List not found: %s', list_path);
    end

    fid = fopen(list_path, 'r');
    if fid < 0
        error('resave_sparse_v73:ListUnreadable', 'Cannot read %s', list_path);
    end
    c = textscan(fid, '%s%s', 'Delimiter', '\t');
    fclose(fid);
    src_list = c{1};
    dst_list = c{2};
    n = numel(src_list);
    fprintf('resave_sparse_v73: %d file(s) to convert\n', n);

    n_ok = 0;
    n_fail = 0;
    for k = 1:n
        src = src_list{k};
        dst = dst_list{k};
        fprintf('[%3d/%3d] %s\n', k, n, src);
        try
            if ~isfile(src)
                error('source not found');
            end
            S = load(src);
            if ~isfield(S, 'metaData') || ~isfield(S, 'frames')
                error('expected variables "metaData" and "frames"');
            end
            % Guard the invariant this whole import rests on: these mats are
            % the mirror cam and are ALREADY flipped. A file without the
            % marker is not what we think it is -- refuse rather than
            % silently import a cam whose orientation disagrees with the
            % calibration.
            if ~isfield(S.metaData, 'isFlipped') || ~S.metaData.isFlipped
                error(['metaData.isFlipped is absent or false -- this mat is ' ...
                       'not a flipped mirror cam; refusing to convert']);
            end
            n_frames = numel(S.frames);

            % Stage next to the destination, commit with movefile, so an
            % interrupted save never leaves a half-written mat under the
            % final name.
            tmp = [dst '.partial'];
            if isfile(tmp)
                delete(tmp);
            end
            save(tmp, '-struct', 'S', '-v7.3');
            movefile(tmp, dst);

            fprintf('          -> %s  (%d frames, isFlipped=%d)\n', ...
                    dst, n_frames, S.metaData.isFlipped);
            n_ok = n_ok + 1;
        catch ME
            fprintf(2, '          ** FAILED: %s\n', ME.message);
            n_fail = n_fail + 1;
        end
    end

    fprintf('\nresave_sparse_v73 done: %d ok, %d failed (of %d)\n', ...
            n_ok, n_fail, n);
    if n_fail > 0
        error('resave_sparse_v73:SomeFailed', '%d file(s) failed', n_fail);
    end
end
