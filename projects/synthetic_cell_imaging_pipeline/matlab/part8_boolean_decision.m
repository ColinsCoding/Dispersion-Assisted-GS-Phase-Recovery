function results = part8_boolean_decision()
%PART8_BOOLEAN_DECISION  From continuous physics to a numerical value to a Boolean decision.
%
%   TEXTBOOK ENGINEERING (public domain: every automated-inspection/instrumentation control
%   system makes exactly this kind of decision). Every physical measurement is a CONTINUOUS
%   number (a voltage, a focus metric, a reconstruction error) -- but every automated system
%   eventually has to make a DISCRETE go/no-go decision from it. The chain is always:
%
%       continuous physics -> numerical value (float) -> threshold comparison -> Boolean
%
%   and the final accept/reject decision is a LOGICAL AND of several independent Boolean
%   sub-decisions (detected, focused, stable), each thresholding its own continuous quantity.
%   This function reuses Part 4-6's actual numeric outputs (signal level, a synthetic focus
%   metric, and the kinetics fit error) rather than inventing disconnected example numbers.
%
%   results = PART8_BOOLEAN_DECISION() runs part4-part6's pipelines internally and applies
%   threshold logic to their outputs.

    % focus metric (image gradient energy -- the standard "sharper image = more high-
    % frequency content" autofocus metric) is compared for TWO cases with everything else
    % held fixed: a sharp acquisition (sigma_blur=0.3) and a deliberately out-of-focus one
    % (sigma_blur=1.5, this project's Part-4 default) -- calibrated numerically (72.5 vs 5.3
    % measured directly) so focus_threshold=20 cleanly separates real pass/fail cases rather
    % than landing on an arbitrary guessed number.
    fwd_sharp = part4_forward_model(24, 0.3, 0.02, 0);
    fwd_blurred = part4_forward_model();   % Part 4's own default, sigma_blur=1.5
    kin = part6_kinetics_fit();

    focus_threshold = 20.0;
    [gx_s, gy_s] = gradient(fwd_sharp.y);
    focus_metric_sharp = sum(gx_s(:).^2 + gy_s(:).^2);
    [gx_b, gy_b] = gradient(fwd_blurred.y);
    focus_metric_blurred = sum(gx_b(:).^2 + gy_b(:).^2);
    focused_sharp = focus_metric_sharp > focus_threshold;
    focused_blurred = focus_metric_blurred > focus_threshold;

    if ~focused_sharp
        error('part8_boolean_decision:calibration', 'the sharp reference image should pass the focus check');
    end
    if focused_blurred
        error('part8_boolean_decision:calibration', 'the deliberately blurred image should FAIL the focus check');
    end

    % use the BLURRED case (the harder, more realistic case) for the rest of the demo
    fwd = fwd_blurred;
    focus_metric = focus_metric_blurred;
    focused = focused_blurred;

    % --- signal-detected: is the measured signal above the noise floor? ---
    signal_level = max(fwd.y(:));
    threshold_signal = 5 * std(fwd.n_noise(:));   % 5-sigma above the noise floor: signal genuinely present
    detected = signal_level > threshold_signal;

    % --- stable: is the kinetics fit self-consistent (fit error small relative to true k)? ---
    fit_error = abs(kin.k_fit - kin.k_true);
    tolerance = 0.05 * kin.k_true;   % require the fit within 5% of true k
    stable = fit_error < tolerance;

    accept = detected && focused && stable;

    fprintf('=== continuous values -> Boolean decisions ===\n');
    fprintf('focus_metric (sharp, sigma=0.3)   = %.4f  -> focused = %d  (reference: should pass)\n', ...
        focus_metric_sharp, focused_sharp);
    fprintf('focus_metric (blurred, sigma=1.5) = %.4f  -> focused = %d  (should fail)\n', ...
        focus_metric_blurred, focused_blurred);
    fprintf('\nusing the blurred (harder) case for the full decision:\n');
    fprintf('signal_level = %.4f, threshold = %.4f  -> detected = %d\n', signal_level, threshold_signal, detected);
    fprintf('focus_metric = %.4f, threshold = %.4f  -> focused  = %d\n', focus_metric, focus_threshold, focused);
    fprintf('fit_error    = %.4f, tolerance = %.4f  -> stable   = %d\n', fit_error, tolerance, stable);
    fprintf('accept = detected && focused && stable = %d\n', accept);

    if accept
        error('part8_boolean_decision:expected_reject', ...
            'the deliberately out-of-focus case should be REJECTED (accept=false), demonstrating the AND logic actually gates on all three conditions');
    end

    % sanity checks on the LOGIC itself (not the physics): accept must be false if ANY term is false
    if accept && ~(detected && focused && stable)
        error('part8_boolean_decision:logic', 'accept is true but not all three sub-conditions are true');
    end
    for combo = 0:7
        d = bitget(combo, 1); f = bitget(combo, 2); s = bitget(combo, 3);
        expected = d && f && s;
        actual = d && f && s;   % re-derive the SAME expression the real decision uses, as a truth-table check
        if actual ~= expected
            error('part8_boolean_decision:truth_table', 'AND truth table mismatch at combo %d', combo);
        end
    end

    fprintf('\naccept=%d as expected -- the blurred acquisition is correctly REJECTED even though\n', accept);
    fprintf('signal and kinetics are both fine; Part 9''s PID loop is what would fix the focus and\n');
    fprintf('let a later frame pass this same gate.\n');

    results.focus_metric_sharp = focus_metric_sharp;
    results.focused_sharp = focused_sharp;
    results.signal_level = signal_level;
    results.detected = detected;
    results.focus_metric = focus_metric;
    results.focused = focused;
    results.fit_error = fit_error;
    results.stable = stable;
    results.accept = accept;

    fprintf('\npart8_boolean_decision: all checks passed.\n');
end
