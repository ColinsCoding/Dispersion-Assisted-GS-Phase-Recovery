function results = part1_field_theory()
%PART1_FIELD_THEORY  Electrostatics field theory: E=-grad(V), div(E), curl(E), Laplacian(V).
%
%   TEXTBOOK PHYSICS (Griffiths-level electrostatics, public domain). Builds the point-charge
%   and physical-dipole potentials symbolically, derives E = -grad(V), and CHECKS (not just
%   asserts) that away from the source: div(E) = 0 (no charge there -> Gauss's law source-free),
%   curl(E) = 0 (E is conservative, which is WHY a scalar potential V can exist for it at all),
%   and laplacian(V) = 0 (V is harmonic there -- the same fact as div(E)=0, from a second angle:
%   Laplacian(V) = div(grad V) = div(-E) = -div(E)).
%
%   GEOMETRY: the point charge is spherically symmetric, so E depends on r alone and points
%   radially -- falls off as 1/r^2 (inverse-square). The dipole breaks that symmetry: two
%   nearly-coincident equal-and-opposite charges mostly cancel except for their small
%   separation a, so the FAR field falls off faster (~1/r^3 on-axis) than either charge's own
%   1/r^2 field alone -- cancellation, not a new physical law, produces the steeper falloff.
%
%   results = PART1_FIELD_THEORY() returns a struct with the symbolic potentials, fields, and
%   divergence/curl/Laplacian results for both configurations.

    syms x y z real
    r = sqrt(x^2 + y^2 + z^2);

    % ---- point charge (spherically symmetric) ----
    k = 1; q = 1;   % Coulomb constant * charge, normalized to 1 for symbolic clarity
    V_point = k * q / r;
    E_point = -gradient(V_point, [x, y, z]);
    div_point = simplify(divergence(E_point, [x, y, z]));
    curl_point = simplify(curl(E_point, [x, y, z]));
    lap_point = simplify(laplacian(V_point, [x, y, z]));

    fprintf('=== Point charge: V = kq/r ===\n');
    fprintf('E = -grad(V):\n');
    disp(E_point);
    fprintf('div(E), away from r=0 (expect 0): %s\n', char(div_point));
    fprintf('curl(E) (expect [0;0;0] -- E is conservative):\n');
    disp(curl_point);
    fprintf('laplacian(V), away from r=0 (expect 0): %s\n', char(lap_point));

    if div_point ~= 0
        error('part1_field_theory:div_point', 'div(E) for a point charge should vanish away from the origin');
    end
    if any(curl_point ~= 0)
        error('part1_field_theory:curl_point', 'curl(E) should vanish identically for any gradient field');
    end
    if lap_point ~= 0
        error('part1_field_theory:lap_point', 'laplacian(V) for a point charge should vanish away from the origin');
    end

    % ---- physical dipole: +q at z=+a/2, -q at z=-a/2 (Coulomb superposition) ----
    a = sym('a', 'positive');
    r_plus  = sqrt(x^2 + y^2 + (z - a/2)^2);
    r_minus = sqrt(x^2 + y^2 + (z + a/2)^2);
    V_dipole = k * q / r_plus - k * q / r_minus;
    E_dipole = -gradient(V_dipole, [x, y, z]);
    div_dipole = simplify(divergence(E_dipole, [x, y, z]));

    fprintf('\n=== Physical dipole: V = kq/r_+ - kq/r_- ===\n');
    fprintf('div(E), away from both charges (expect 0): %s\n', char(div_dipole));

    if div_dipole ~= 0
        error('part1_field_theory:div_dipole', 'div(E) for a dipole should vanish away from both charges');
    end

    % on-axis far-field falloff check: numerically compare 1/r^2 (point) vs 1/r^3 (dipole)
    z_far = [5, 10, 20, 40];
    Ez_point_fn = matlabFunction(E_point(3), 'Vars', [x, y, z]);
    Ez_dipole_fn = matlabFunction(subs(E_dipole(3), a, 0.1), 'Vars', [x, y, z]);
    Ez_point_vals = arrayfun(@(zz) Ez_point_fn(0, 0, zz), z_far);
    Ez_dipole_vals = arrayfun(@(zz) Ez_dipole_fn(0, 0, zz), z_far);

    ratio_point = log(abs(Ez_point_vals(1) / Ez_point_vals(end))) / log(z_far(end) / z_far(1));
    ratio_dipole = log(abs(Ez_dipole_vals(1) / Ez_dipole_vals(end))) / log(z_far(end) / z_far(1));
    fprintf('\nFar-field power-law falloff (fit exponent n in E ~ 1/r^n):\n');
    fprintf('  point charge:  n = %.3f  (expect ~2)\n', ratio_point);
    fprintf('  dipole (a=0.1): n = %.3f  (expect ~3)\n', ratio_dipole);

    if abs(ratio_point - 2) > 0.05
        error('part1_field_theory:falloff_point', 'point-charge field should fall off as 1/r^2');
    end
    if abs(ratio_dipole - 3) > 0.1
        error('part1_field_theory:falloff_dipole', 'far-field dipole should fall off as ~1/r^3 on-axis');
    end

    results.V_point = V_point;
    results.E_point = E_point;
    results.div_point = div_point;
    results.curl_point = curl_point;
    results.lap_point = lap_point;
    results.V_dipole = V_dipole;
    results.E_dipole = E_dipole;
    results.div_dipole = div_dipole;
    results.falloff_exponent_point = ratio_point;
    results.falloff_exponent_dipole = ratio_dipole;

    fprintf('\npart1_field_theory: all checks passed.\n');
end
