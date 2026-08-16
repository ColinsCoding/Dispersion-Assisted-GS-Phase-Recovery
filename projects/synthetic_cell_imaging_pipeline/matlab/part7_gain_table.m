function results = part7_gain_table(P_opt_W)
%PART7_GAIN_TABLE  Subsystem input/output/gain/loss/units bookkeeping through a full imaging chain.
%
%   TEXTBOOK ENGINEERING (public domain: every optoelectronics/instrumentation textbook's
%   signal-chain analysis). Every subsystem in an imaging chain has a well-defined
%   INPUT quantity, OUTPUT quantity, a GAIN (or loss) relating them, and units that must
%   compose correctly end-to-end -- get any one wrong and the numbers downstream are
%   meaningless even if every individual formula is "right". This function walks one
%   concrete numeric example, in real units, all the way through:
%
%       optical power -> photocurrent -> amplifier voltage -> ADC count -> recovered power
%
%   and CHECKS (not just states) that composing all the individual gains reproduces the
%   directly-computed end-to-end gain, and that converting the ADC count back through the
%   chain recovers the original optical power to within quantization error.
%
%   results = PART7_GAIN_TABLE(P_opt_W) with default P_opt_W = 50e-6 (50 uW).

    if nargin < 1 || isempty(P_opt_W), P_opt_W = 50e-6; end

    % subsystem parameters (illustrative, not tied to any specific commercial part)
    responsivity_A_per_W = 0.8;      % photodiode: A/W
    Rf_ohm = 2e4;                     % transimpedance amplifier feedback resistor: V/A
    n_bits = 12;                      % ADC resolution
    V_ref = 3.3;                      % ADC full-scale reference voltage

    % ---- walk the chain forward ----
    I_pd_A = responsivity_A_per_W * P_opt_W;
    V_amp_V = Rf_ohm * I_pd_A;
    adc_count = round((V_amp_V / V_ref) * (2^n_bits - 1));
    adc_count = min(max(adc_count, 0), 2^n_bits - 1);   % clip to representable range

    % ---- invert the chain: ADC count -> recovered optical power ----
    V_amp_recovered = (adc_count / (2^n_bits - 1)) * V_ref;
    I_pd_recovered = V_amp_recovered / Rf_ohm;
    P_opt_recovered = I_pd_recovered / responsivity_A_per_W;

    recon_error_frac = abs(P_opt_recovered - P_opt_W) / P_opt_W;
    quantization_step_W = V_ref / (2^n_bits - 1) / Rf_ohm / responsivity_A_per_W;

    fprintf('=== signal chain: %.3g W optical input ===\n', P_opt_W);
    fprintf('photocurrent      I_pd  = %.4e A\n', I_pd_A);
    fprintf('amplifier voltage V_amp = %.4f V\n', V_amp_V);
    fprintf('ADC count (of %d)       = %d\n', 2^n_bits - 1, adc_count);
    fprintf('recovered optical power = %.4e W (true %.4e W, error %.2f%%, quantization step %.3e W)\n', ...
        P_opt_recovered, P_opt_W, 100 * recon_error_frac, quantization_step_W);

    if recon_error_frac > 0.02 && abs(P_opt_recovered - P_opt_W) > 3 * quantization_step_W
        error('part7_gain_table:round_trip', 'recovered power should match the input to within a few quantization steps');
    end

    % ---- build the gain/loss/units table ----
    subsystem = {'photodiode'; 'transimpedance amp'; 'ADC'; 'ADC vector -> recovered power'};
    input_qty = {'optical power P (W)'; 'photocurrent I_pd (A)'; 'amplifier voltage V_amp (V)'; 'ADC count (LSB)'};
    output_qty = {'photocurrent I_pd (A)'; 'amplifier voltage V_amp (V)'; 'ADC count (LSB)'; 'recovered power P_hat (W)'};
    gain_value = [responsivity_A_per_W; Rf_ohm; (2^n_bits - 1) / V_ref; 1 / ((2^n_bits - 1) / V_ref * Rf_ohm * responsivity_A_per_W)];
    gain_units = {'A/W'; 'V/A'; 'LSB/V'; 'W/LSB'};

    T = table(subsystem, input_qty, output_qty, gain_value, gain_units);
    disp(T);

    % end-to-end gain: compose the individual stage gains and cross-check against the
    % directly-computed input/output ratio (skipping quantization, which is not a smooth gain)
    end_to_end_gain_composed = responsivity_A_per_W * Rf_ohm;   % W -> V, before the ADC's digitization
    end_to_end_gain_direct = V_amp_V / P_opt_W;
    composition_error = abs(end_to_end_gain_composed - end_to_end_gain_direct) / end_to_end_gain_direct;
    fprintf('\nend-to-end analog gain (P_opt -> V_amp): composed = %.4f V/W, direct = %.4f V/W (error %.2e)\n', ...
        end_to_end_gain_composed, end_to_end_gain_direct, composition_error);

    if composition_error > 1e-9
        error('part7_gain_table:composition', 'composed subsystem gains do not reproduce the direct end-to-end gain');
    end

    results.table = T;
    results.P_opt_W = P_opt_W;
    results.I_pd_A = I_pd_A;
    results.V_amp_V = V_amp_V;
    results.adc_count = adc_count;
    results.P_opt_recovered = P_opt_recovered;
    results.recon_error_frac = recon_error_frac;
    results.end_to_end_gain_V_per_W = end_to_end_gain_direct;

    fprintf('\npart7_gain_table: all checks passed.\n');
end
