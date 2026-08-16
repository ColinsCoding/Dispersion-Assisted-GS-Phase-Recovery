function summary = part14_integrated_project()
%PART14_INTEGRATED_PROJECT  Final integrated project: every earlier part, one coherent run.
%
%   synthetic cell -> optical response (forward model, Part 4) -> detector (gain/noise,
%   Part 4/7) -> noisy samples -> matrix reconstruction (Part 5) -> kinetic-parameter
%   estimate (Part 6) -> Boolean classification (Part 8) -> PID-style focus/alignment
%   simulation (Part 9).
%
%   This is THIN ORCHESTRATION -- it calls the already-verified functions from Parts 4-9
%   and Part 11's pipeline, and cross-checks that their outputs are still mutually
%   consistent when run together, rather than re-implementing any of their math here.
%
%   summary = PART14_INTEGRATED_PROJECT() runs the full chain and returns one struct
%   collecting every stage's key numbers.

    here = fileparts(mfilename('fullpath'));
    pipeline_dir = fullfile(here, 'part11_pipeline');
    if isempty(which('run_pipeline')), addpath(pipeline_dir); end

    fprintf('##############################################################\n');
    fprintf('# PART 14: FINAL INTEGRATED PROJECT\n');
    fprintf('##############################################################\n\n');

    fprintf('--- stage 1-5: acquire -> preprocess -> calibrate -> reconstruct -> classify ---\n');
    pipeline_results = run_pipeline();

    fprintf('\n--- stage 6: matrix-analysis cross-check (rank/SVD/condition number of the SAME H) ---\n');
    matrix_results = part5_matrix_analysis();

    fprintf('\n--- stage 7: kinetic-parameter estimate (independent synthetic dataset) ---\n');
    kinetics_results = part6_kinetics_fit();

    fprintf('\n--- stage 8: Boolean classification (raw-vs-reconstructed contrast) ---\n');
    boolean_results = part8_boolean_decision();

    fprintf('\n--- stage 9: PID-style focus/alignment simulation ---\n');
    pid_results = part9_pid_feedback();

    % ---- cross-consistency checks: do the independently-run parts agree where they should? ----
    if matrix_results.cond_H < 1e6
        error('part14_integrated_project:inconsistent', ...
            'expected the SAME ill-conditioned H as Part 5 to be reused/reproduced by Part 11''s pipeline');
    end
    if pipeline_results.reconstruction_error > 0.6
        error('part14_integrated_project:reconstruction', 'integrated pipeline reconstruction error unexpectedly large');
    end
    if ~pipeline_results.decision.accept
        error('part14_integrated_project:classification', 'integrated pipeline should PASS classification (reconstructed, in-focus case)');
    end
    if boolean_results.accept
        error('part14_integrated_project:boolean_contrast', ...
            'Part 8''s raw-blur case should still FAIL, contrasting with the reconstructed pipeline''s PASS');
    end
    if pid_results.final_error > 0.15 * abs(pid_results.error_hist(1))
        error('part14_integrated_project:pid', 'PID loop did not converge sufficiently within the integrated run');
    end

    summary.forward_model_N = pipeline_results.params.N;
    summary.cond_H = matrix_results.cond_H;
    summary.rank_H = matrix_results.rank_H;
    summary.pipeline_reconstruction_error = pipeline_results.reconstruction_error;
    summary.pipeline_decision = pipeline_results.decision.label;
    summary.raw_blur_decision = boolean_results.accept;
    summary.k_true = kinetics_results.k_true;
    summary.k_fit = kinetics_results.k_fit;
    summary.pid_final_error = pid_results.final_error;
    summary.pid_settled_step = pid_results.settled_idx;

    fprintf('\n##############################################################\n');
    fprintf('# INTEGRATED SUMMARY\n');
    fprintf('##############################################################\n');
    fprintf('forward model: N=%d, H is %dx%d, cond(H)=%.3e, rank(H)=%d\n', ...
        summary.forward_model_N, summary.forward_model_N^2, summary.forward_model_N^2, summary.cond_H, summary.rank_H);
    fprintf('reconstructed-pipeline decision: %s (reconstruction rel. error %.4f)\n', ...
        summary.pipeline_decision, summary.pipeline_reconstruction_error);
    fprintf('raw-blur-only decision (Part 8 contrast): accept=%d\n', summary.raw_blur_decision);
    fprintf('kinetics: k_true=%.4f, k_fit=%.4f (%.2f%% error)\n', ...
        summary.k_true, summary.k_fit, 100 * abs(summary.k_fit - summary.k_true) / summary.k_true);
    fprintf('PID: final error=%.4f, settled by step %d\n', summary.pid_final_error, summary.pid_settled_step);
    fprintf('\npart14_integrated_project: all cross-consistency checks passed.\n');
end
