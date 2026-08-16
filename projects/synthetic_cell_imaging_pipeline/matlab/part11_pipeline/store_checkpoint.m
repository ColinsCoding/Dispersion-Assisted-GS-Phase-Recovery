function store_checkpoint(checkpoint_path, data)
%STORE_CHECKPOINT  Stage 6: save pipeline results to a .mat checkpoint file.
%
%   A plain data-persistence step -- writes the given struct to disk with save(). This is
%   NOT automation of any lab hardware or laser: it only ever touches a local .mat file on
%   disk, nothing physical.
%
%   STORE_CHECKPOINT(checkpoint_path, data).

    save(checkpoint_path, 'data');
    info = dir(checkpoint_path);
    fprintf('[store_checkpoint] saved %s (%d bytes)\n', checkpoint_path, info.bytes);
end
