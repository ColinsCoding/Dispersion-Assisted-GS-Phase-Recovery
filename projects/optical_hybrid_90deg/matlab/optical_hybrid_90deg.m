function [output0, output90, output180, output270] = optical_hybrid_90deg( ...
    signal_input, ...
    local_oscillator, ...
    insertion_loss_signal, ...
    insertion_loss_local_oscillator, ...
    phase_imbalance_slo, ...
    phase_imbalance_iq, ...
    insertion_loss_imbalance_i, ...
    insertion_loss_imbalance_q ...
)
    % Simulates a 90-degree Optical Hybrid with given parameters.
    % Parameters:
    %   signal_input (complex): Input optical signal (E-field).
    %   local_oscillator (complex): Local oscillator signal (E-field).
    %   insertion_loss_signal (float): Signal input insertion loss in dB.
    %   insertion_loss_local_oscillator (float): Local oscillator insertion loss in dB.
    %   phase_imbalance_slo (float): Phase imbalance (signal and local oscillator) in radians.
    %   phase_imbalance_iq (float): Phase imbalance (I and Q branches) in radians.
    %   insertion_loss_imbalance_i (float): Insertion loss imbalance for I branch in dB.
    %   insertion_loss_imbalance_q (float): Insertion loss imbalance for Q branch in dB.
    % Returns:
    %   output0, output90, output180, output270: Complex numbers representing the output signals with 0°, 90°, 180°, and 270° phase differences.

    % Convert dB parameters to linear scale
    IL_signal = 10 ^ (-insertion_loss_signal / 10);
    IL_lo = 10 ^ (-insertion_loss_local_oscillator / 10);
    Imb_I = 10 ^ (-insertion_loss_imbalance_i / 10);
    Imb_Q = 10 ^ (-insertion_loss_imbalance_q / 10);

    % Adjust input signals based on insertion loss
    signal_input = signal_input * IL_signal;
    local_oscillator = local_oscillator * IL_lo;

    % Transmission matrix for the ideal case
    T = [
        1, 1;
        1, -1;
        1j, 1j;
        1j, -1j
    ];

    % Adjust transmission matrix for phase imbalances
    phase_matrix = diag([
        1;
        exp(1j * phase_imbalance_slo);
        exp(1j * phase_imbalance_iq);
        exp(1j * (phase_imbalance_slo + phase_imbalance_iq))
    ]);

    % Adjust for insertion loss imbalances
    imbalance_matrix = diag([1; Imb_I; Imb_Q; Imb_I * Imb_Q]);

    % Combine adjustments into the final transmission matrix
    T_adjusted = imbalance_matrix * phase_matrix * T;

    % Create input signal vector
    E_in = [signal_input; local_oscillator];

    % Calculate output signals
    E_out = T_adjusted * E_in;

    % Return the outputs with specified phase differences
    output0 = E_out(1);
    output90 = E_out(2);
    output180 = E_out(3);
    output270 = E_out(4);
end