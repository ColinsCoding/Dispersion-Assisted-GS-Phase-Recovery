function y_calibrated = calibrate(y_processed, gain_map)
%CALIBRATE  Stage 3: divide out the known per-pixel sensitivity (gain) map.
%
%   Every real sensor's pixels don't all respond identically to the same light level; a
%   calibration step (typically measured once against a uniform reference source, a "flat
%   field") divides that fixed pattern back out before reconstruction sees the data.
%
%   y_calibrated = CALIBRATE(y_processed, gain_map).

    if any(gain_map(:) <= 0)
        error('calibrate:bad_gain_map', 'gain_map must be strictly positive (it is a sensitivity multiplier)');
    end
    y_calibrated = y_processed ./ gain_map;
    fprintf('[calibrate] gain map range [%.3f, %.3f] divided out\n', min(gain_map(:)), max(gain_map(:)));
end
