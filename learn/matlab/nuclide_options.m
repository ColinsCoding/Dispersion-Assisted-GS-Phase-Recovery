% nuclide_options.m -- protons / neutrons / electrons from ^A X^q, and a loop
% that tests every multiple-choice option. MATLAB twin of dgs/nuclide_notation.py.
%
%   protons   Z = atomic number of X
%   neutrons  N = A - Z
%   electrons   = Z - q        (a negative ion carries EXTRA electrons)
%
% Run headless:  matlab -batch "run('learn/matlab/nuclide_options.m')"

Zof = containers.Map({'H','He','C','N','O','F','Na','Cl','Fe','U'}, ...
                     {  1,   2,  6,  7,  8,  9,  11,  17,  26, 92});

notation = '16O2-';
tok = regexp(notation, '^(\d+)([A-Z][a-z]?)(\d*)([+-]?)$', 'tokens', 'once');
A   = str2double(tok{1});
Z   = Zof(tok{2});
q   = 0;
if ~isempty(tok{4})
    mag = 1;
    if ~isempty(tok{3}), mag = str2double(tok{3}); end
    q = mag * (2 * strcmp(tok{4}, '+') - 1);
end
N = A - Z;
e = Z - q;
fprintf('%s: A = %d, Z = %d, q = %+d  ->  %d protons, %d neutrons, %d electrons\n', ...
        notation, A, Z, q, Z, N, e);

% options from the Mastering question (NaN row = "None of these / other")
opts = [ 8  8 10;
         8  8  2;
       NaN NaN NaN;
        16 16 18;
        16 16  2;
        16  0 18];

nCorrect = 0;
for i = 1:size(opts, 1)
    o = opts(i, :);
    if any(isnan(o))
        fprintf('option %d: (not numeric)\n', i);
        continue
    end
    ok = isequal(o, [Z N e]);
    nCorrect = nCorrect + ok;
    fprintf('option %d: (%2d, %2d, %2d)  %s\n', i, o, string(ok));
end

assert(nCorrect == 1, 'exactly one option should match');
assert(isequal(opts(1, :), [Z N e]), 'option 1 should be the correct one');
disp('MATLAB checks passed');
