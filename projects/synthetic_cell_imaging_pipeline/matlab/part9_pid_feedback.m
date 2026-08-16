function results = part9_pid_feedback(Kp, Ki, Kd, target, n_steps)
%PART9_PID_FEEDBACK  A closed-loop autofocus/alignment simulation driven by a discrete PID controller.
%
%   TEXTBOOK CONTROL THEORY (public domain: PID control is one of the oldest, most widely
%   taught feedback-control algorithms, in every controls textbook). error = target -
%   measured; the controller outputs
%
%       u[n] = Kp*e[n] + Ki*sum(e[0..n]) + Kd*(e[n]-e[n-1]),
%
%   a weighted sum of the CURRENT error (proportional), its ACCUMULATED history (integral --
%   removes steady-state offset), and its RATE of change (derivative -- damps overshoot).
%
%   OUR EDUCATIONAL SIMULATION: a simple 1D "focus position" plant. Moving a lens/stage by
%   u[n] changes a scalar focus-error signal by a fixed PLANT GAIN (arbitrary units) plus
%   process noise -- not a model of any specific real autofocus mechanism, just enough
%   dynamics to make PID's proportional/integral/derivative terms all matter.
%
%   results = PART9_PID_FEEDBACK(Kp, Ki, Kd, target, n_steps) with defaults Kp=0.6, Ki=0.15,
%   Kd=0.05, target=0 (drive the focus error to zero), n_steps=60.

    if nargin < 1 || isempty(Kp), Kp = 0.6; end
    if nargin < 2 || isempty(Ki), Ki = 0.15; end
    if nargin < 3 || isempty(Kd), Kd = 0.05; end
    if nargin < 4 || isempty(target), target = 0; end
    if nargin < 5 || isempty(n_steps), n_steps = 60; end

    rng(2);
    plant_gain = 0.8;                 % how much one unit of control moves the measured quantity
    process_noise_std = 0.01;
    dt = 1;

    measured = zeros(n_steps, 1);
    reference = target * ones(n_steps, 1);
    error_hist = zeros(n_steps, 1);
    control_hist = zeros(n_steps, 1);

    measured(1) = 3.5;                % start badly out of focus (large initial error)
    integral_term = 0;
    prev_error = target - measured(1);

    for n = 1:n_steps
        e = target - measured(n);
        integral_term = integral_term + e * dt;
        derivative_term = (e - prev_error) / dt;

        u = Kp * e + Ki * integral_term + Kd * derivative_term;

        error_hist(n) = e;
        control_hist(n) = u;
        prev_error = e;

        if n < n_steps
            measured(n + 1) = measured(n) + plant_gain * u * dt + process_noise_std * randn();
        end
    end

    final_error = abs(error_hist(end));
    settled_idx = find(abs(error_hist) > 0.05 * abs(error_hist(1)), 1, 'last');
    if isempty(settled_idx), settled_idx = 1; end

    fprintf('=== PID autofocus/alignment simulation ===\n');
    fprintf('Kp=%.3f, Ki=%.3f, Kd=%.3f, target=%.3f, initial measured=%.3f\n', Kp, Ki, Kd, target, measured(1));
    fprintf('final error (step %d) = %.4f  (started at %.4f)\n', n_steps, final_error, abs(error_hist(1)));
    fprintf('settled to within 5%% of initial error by step %d\n', settled_idx);

    if final_error > 0.15 * abs(error_hist(1))
        error('part9_pid_feedback:convergence', 'PID loop did not sufficiently reduce the error by the end of the simulation');
    end
    if abs(error_hist(end)) >= abs(error_hist(1))
        error('part9_pid_feedback:no_progress', 'PID loop made no net progress toward the target');
    end

    fig = figure('Visible', 'off');
    subplot(3, 1, 1); plot(1:n_steps, reference, 'k--', 1:n_steps, measured, 'b-', 'LineWidth', 1.3);
    legend('reference', 'measured'); ylabel('focus position'); grid on;
    title('PID autofocus/alignment: reference vs. measured');
    subplot(3, 1, 2); plot(1:n_steps, error_hist, 'r-', 'LineWidth', 1.3);
    ylabel('error = target - measured'); grid on;
    subplot(3, 1, 3); plot(1:n_steps, control_hist, 'm-', 'LineWidth', 1.3);
    xlabel('step n'); ylabel('control signal u[n]'); grid on;
    out_dir = fileparts(mfilename('fullpath'));
    saveas(fig, fullfile(out_dir, 'part9_pid_feedback.png'));
    close(fig);

    results.reference = reference;
    results.measured = measured;
    results.error_hist = error_hist;
    results.control_hist = control_hist;
    results.final_error = final_error;
    results.settled_idx = settled_idx;

    fprintf('\npart9_pid_feedback: all checks passed. Figure saved to part9_pid_feedback.png\n');
end
