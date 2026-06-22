function save_optical_hybrid_outputs()
    % Example usage
    signal_input = 1 + 1j;  % Example complex signal input
    local_oscillator = 1 - 1j;  % Example complex local oscillator signal

    [output0, output90, output180, output270] = optical_hybrid_90deg( ...
        signal_input, ...
        local_oscillator, ...
        1.0, ... % insertion_loss_signal
        0.5, ... % insertion_loss_local_oscillator
        pi / 18, ... % phase_imbalance_slo (10 degrees in radians)
        pi / 36, ... % phase_imbalance_iq (5 degrees in radians)
        0.2, ... % insertion_loss_imbalance_i
        0.1 ... % insertion_loss_imbalance_q
    );

    % Save the outputs to a CSV file
    outputs = [output0; output90; output180; output270];
    output_table = table(real(outputs), imag(outputs), 'VariableNames', {'Real', 'Imaginary'});
    writetable(output_table, '../data/optical_hybrid_outputs.csv');

    disp('Outputs saved to optical_hybrid_outputs.csv');
end