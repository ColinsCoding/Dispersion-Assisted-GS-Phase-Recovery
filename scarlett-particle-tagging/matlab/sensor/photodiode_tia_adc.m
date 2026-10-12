function out = photodiode_tia_adc(optical_power, cfg)
%PHOTODIODE_TIA_ADC  Behavioural sensor chain: photodiode -> TIA -> PGA -> ADC.
%
%   out = PHOTODIODE_TIA_ADC(optical_power)
%   out = PHOTODIODE_TIA_ADC(optical_power, cfg)
%
%   optical_power  optical power on the detector per channel [W]
%   cfg            struct, see default fields below (nominal DESIGN
%                  ASSUMPTIONS for simulation, not measured hardware values)
%
%   Model (noiseless expectation plus an RMS noise budget):
%       i      = responsivity * P + dark_current                       [A]
%       v      = i * tia_gain                                          [V]
%       gain   = largest PGA setting with max(v)*gain <= 0.8*adc_vref
%       code   = round(v * gain / LSB),  LSB = adc_vref / 2^adc_bits
%       B      = (pi/2) * antialias_cutoff      noise bandwidth of one pole
%       shot      = sqrt(2 q i B)               [A]
%       thermal   = sqrt(4 kB T B / tia_gain)   [A]
%       amplifier = amplifier_current_noise * sqrt(B)   [A]
%
%   Output struct: current [A], volts [V], gain, codes, clipped,
%   noise_shot, noise_thermal, noise_amplifier, noise_total [A],
%   snr (signal current / noise_total), lsb [V].

    if nargin < 2
        cfg = struct();
    end
    defaults = struct('responsivity', 1.0, 'dark_current', 1e-9, 'tia_gain', 1e6, ...
        'amplifier_current_noise', 1e-13, 'temperature', 293.15, ...
        'pga_gains', [1 2 4 8 16 32 64 128], 'antialias_cutoff', 40e3, ...
        'adc_bits', 12, 'adc_vref', 1.0, 'sampling_rate', 100e3);
    names = fieldnames(defaults);
    for i = 1:numel(names)
        if ~isfield(cfg, names{i})
            cfg.(names{i}) = defaults.(names{i});
        end
    end
    if any(optical_power(:) < 0)
        error('photodiode_tia_adc:input', 'optical_power must be non-negative [W].');
    end
    if cfg.antialias_cutoff >= cfg.sampling_rate/2
        error('photodiode_tia_adc:nyquist', 'Anti-alias cutoff must be below sampling_rate/2.');
    end

    q_e = 1.602176634e-19;      % elementary charge [C]
    k_b = 1.380649e-23;         % Boltzmann constant [J/K]

    signal_current = cfg.responsivity*optical_power;
    out.current = signal_current + cfg.dark_current;
    out.volts = out.current*cfg.tia_gain;

    usable = cfg.pga_gains(max(out.volts(:))*cfg.pga_gains <= 0.8*cfg.adc_vref);
    if isempty(usable)
        out.gain = min(cfg.pga_gains);
    else
        out.gain = max(usable);
    end
    out.lsb = cfg.adc_vref/2^cfg.adc_bits;
    raw = round(out.volts*out.gain/out.lsb);
    out.clipped = raw < 0 | raw > 2^cfg.adc_bits - 1;
    out.codes = min(max(raw, 0), 2^cfg.adc_bits - 1);

    bandwidth = pi/2*cfg.antialias_cutoff;
    out.noise_shot = sqrt(2*q_e*out.current*bandwidth);
    out.noise_thermal = sqrt(4*k_b*cfg.temperature*bandwidth/cfg.tia_gain);
    out.noise_amplifier = cfg.amplifier_current_noise*sqrt(bandwidth);
    out.noise_total = sqrt(out.noise_shot.^2 + out.noise_thermal^2 + out.noise_amplifier^2);
    out.snr = signal_current./out.noise_total;
end
