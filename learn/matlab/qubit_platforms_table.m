% qubit_platforms_table.m -- the qubit-drive table computed independently in MATLAB from the SAME
% JSON file the Python module reads (dgs/qubit_platforms.json).
%
% In plain words: the numbers for the transmon, the CSUS coupled-LC bench analog and the toy
% electron-in-a-box live in one file. Python (dgs/vector_potential_qubit.py) and this script each
% compute the derived table with their own code, and print it in the same "platform.key value" format,
% so scripts/check_qubit_platforms_everywhere.sh can diff them line by line.
%
% Run headless from the repo root:
%   matlab -batch "run('learn/matlab/qubit_platforms_table.m')"

here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(here));
P = jsondecode(fileread(fullfile(root, 'dgs', 'qubit_platforms.json')));
c = P.constants;
e = c.e_C; h = c.h_Js; kB = c.kB_JpK; me = c.m_e_kg;
hbar = h / (2*pi);
out = struct();

% ---- transmon: charge drive, Omega = 2 e beta V0 <0|n|1> / hbar
p = P.platforms.transmon;
EC = p.EC_over_h_Hz * h;  EJ = p.EJ_over_EC * EC;
f01 = (sqrt(8*EJ*EC) - EC) / h;  w01 = 2*pi*f01;
n01 = (EJ/(8*EC))^0.25 / sqrt(2);
V0 = sqrt(2 * 1e-3 * 10^(p.drive_dBm_at_qubit/10) * p.Z0_ohm);
Om = 2*e*p.beta_Cg_over_Csum*V0*n01 / hbar;
out.transmon = struct('f_transition_Hz', f01, 'Csum_F', e^2/(2*EC), 'n01', n01, 'V0_V', V0, ...
    'Omega_rad_s', Om, 'f_Rabi_Hz', Om/(2*pi), 't_pi_s', pi/Om, 'vector_potential_amp', V0/w01, ...
    'n_thermal', 1/expm1(h*f01/(kB*p.T_K)));

% ---- CSUS coupled-LC analog: Omega = k omega0
p = P.platforms.csus_lc_analog;
w0 = 1/sqrt(p.L_H*(p.C_F + p.C_probe_F));
Om = p.k * w0;
out.csus_lc_analog = struct('f_transition_Hz', w0/(2*pi), 'Omega_rad_s', Om, 'f_Rabi_Hz', Om/(2*pi), ...
    't_pi_s', pi/Om, 'vector_potential_amp', p.k*p.L_H, 'n_thermal', 1/expm1(h*w0/(2*pi)/(kB*p.T_K)));

% ---- electron in a box: length gauge q E0 x12 vs velocity gauge (q/m) A0 p12
p = P.platforms.box_atom;
w12 = 3*pi^2*hbar/(2*me*p.a_m^2);
x12 = 16*p.a_m/(9*pi^2);
Om = e*p.E0_Vpm*x12/hbar;
A0 = p.E0_Vpm/w12;
out.box_atom = struct('f_transition_Hz', w12/(2*pi), 'x12_m', x12, 'Omega_rad_s', Om, ...
    'Omega_velocity_gauge_rad_s', (e/me)*A0*(me*w12*x12)/hbar, 'f_Rabi_Hz', Om/(2*pi), 't_pi_s', pi/Om, ...
    'vector_potential_amp', A0, 'n_thermal', 1/expm1(h*w12/(2*pi)/(kB*p.T_K)));

% ---- print in the shared format, sorted like Python
plats = sort(fieldnames(out));
for i = 1:numel(plats)
    s = out.(plats{i});
    keys = sort(fieldnames(s));
    for j = 1:numel(keys)
        fprintf('%s.%s %.12e\n', plats{i}, keys{j}, s.(keys{j}));
    end
end
