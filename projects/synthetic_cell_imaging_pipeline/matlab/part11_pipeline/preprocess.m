function y_processed = preprocess(y_raw)
%PREPROCESS  Stage 2: clip nonphysical negative detector readings.
%
%   A real detector cannot report negative photon counts/current -- any negative values in
%   y_raw come from additive measurement noise dipping below zero at low signal, and must be
%   clipped before any further processing (dividing by, or fitting against, values a real
%   sensor could never produce would just introduce artifacts of its own).
%
%   y_processed = PREPROCESS(y_raw).

    n_clipped = sum(y_raw(:) < 0);
    y_processed = max(y_raw, 0);
    fprintf('[preprocess] clipped %d/%d negative samples\n', n_clipped, numel(y_raw));
end
