function decision = classify(x_hat, y_calibrated, noise_std, focus_threshold)
%CLASSIFY  Stage 5: continuous reconstruction -> Boolean pass/fail decision.
%
%   Same continuous-value -> threshold -> Boolean pattern as Part 8, applied here to the
%   PIPELINE's own reconstructed image x_hat (sharper than the raw blurred measurement, so
%   it uses its own calibrated focus_threshold, not Part 8's).
%
%   decision = CLASSIFY(x_hat, y_calibrated, noise_std, focus_threshold) with default
%   focus_threshold = 30 -- calibrated numerically against this project's own default
%   reconstruction (measures ~40.3 for the regularized reconstruction vs. ~5.3 for the raw
%   blurred measurement Part 8 classifies, see part8_boolean_decision.m), so a properly
%   reconstructed frame PASSES here even though the raw acquisition alone would not.

    if nargin < 4 || isempty(focus_threshold), focus_threshold = 30.0; end

    [gx, gy] = gradient(x_hat);
    focus_metric = sum(gx(:).^2 + gy(:).^2);
    signal_level = max(y_calibrated(:));
    threshold_signal = 5 * noise_std;

    detected = signal_level > threshold_signal;
    focused = focus_metric > focus_threshold;
    accept = detected && focused;

    decision.focus_metric = focus_metric;
    decision.signal_level = signal_level;
    decision.detected = detected;
    decision.focused = focused;
    decision.accept = accept;
    if accept
        decision.label = 'PASS';
    else
        decision.label = 'FAIL';
    end

    fprintf('[classify] signal=%.4f (thr %.4f) detected=%d, focus=%.4f (thr %.4f) focused=%d -> %s\n', ...
        signal_level, threshold_signal, detected, focus_metric, focus_threshold, focused, decision.label);
end
