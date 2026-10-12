function stats = wifi_building3d(wallsCsv, router, fGHz, gifPath)
% WIFI_BUILDING3D  Vectorized 3-D multi-wall WiFi model (MATLAB twin of
% dgs/wifi_building3d.py). Same style: no per-cell loops -- implicit
% expansion builds the N-by-M wall-crossing matrix in one shot.
%
%   stats = wifi_building3d('walls.csv', [10 3 6], 2.4, 'hallway.gif')
%
% wallsCsv columns: x1,y1,x2,y2,loss24,loss5 (exported by Python, single source
% of truth). Prints per-floor coverage / median / p10 and, if gifPath is given,
% writes an animated GIF of the floor the router is on as it walks the hallway.

W = 20; H = 14; nFloors = 3; res = 0.25; fh = 3.5; rxH = 1.0;
Ptx = 20; c = 299792458; thr = -67;
band = 1 + (fGHz >= 4);
slab = [18 25];

T = readmatrix(wallsCsv);
A = T(:, 1:2); B = T(:, 3:4); L = T(:, 4 + band);

xs = res/2:res:W; ys = res/2:res:H;
[X, Y] = meshgrid(xs, ys);
P = []; F = [];
for k = 0:nFloors-1
    P = [P; X(:), Y(:), repmat(k*fh + rxH, numel(X), 1)]; %#ok<AGROW>
    F = [F; repmat(k, numel(X), 1)];                        %#ok<AGROW>
end

Pr = model(router, P, F);
stats = zeros(nFloors, 3);
fprintf('router (%.1f, %.1f, %.1f) m at %.1f GHz\n', router, fGHz);
for k = 0:nFloors-1
    v = Pr(F == k);
    stats(k+1, :) = [100*mean(v >= thr), median(v), prctile(v, 10)];
    fprintf('floor %d: coverage %6.2f %%   median %7.2f dBm   p10 %7.2f dBm\n', k, stats(k+1, :));
end

if nargin >= 4 && ~isempty(gifPath)
    nFrames = 24; path = linspace(5, 13.5, nFrames);
    fl = floor(router(3) / fh);
    sel = F == fl;
    cmap = turbo(256);
    for i = 1:nFrames
        r = [path(i) 3 router(3)];
        v = model(r, P(sel, :), F(sel));
        img = reshape(v, numel(ys), numel(xs));
        idx = uint8(round(255 * min(max((img + 90) / 60, 0), 1)));    % -90..-30 dBm
        idx = flipud(idx);                                             % north up
        if i == 1
            imwrite(idx, cmap, gifPath, 'gif', 'LoopCount', Inf, 'DelayTime', 0.15);
        else
            imwrite(idx, cmap, gifPath, 'gif', 'WriteMode', 'append', 'DelayTime', 0.15);
        end
    end
    fprintf('wrote %s (%d frames)\n', gifPath, nFrames);
end

    function pr = model(r, pts, flr)
        d = sqrt(sum((pts - r).^2, 2));
        fspl = 20*log10(4*pi*max(d, 0.1)*fGHz*1e9/c);
        p = r(1:2); q = pts(:, 1:2);
        cr = @(o, a, b) (a(:,1,:) - o(:,1,:)).*(b(:,2,:) - o(:,2,:)) - (a(:,2,:) - o(:,2,:)).*(b(:,1,:) - o(:,1,:));
        a3 = reshape(A', 1, 2, []); b3 = reshape(B', 1, 2, []);      % 1 x 2 x M
        d1 = sign(cr(p, q, a3)); d2 = sign(cr(p, q, b3));            % N x 1 x M
        d3 = sign(cr(a3, b3, p)); d4 = sign(cr(a3, b3, q));
        crosses = squeeze((d1.*d2 < 0) & (d3.*d4 < 0));               % N x M
        wl = double(crosses) * L;
        slabs = abs(flr - floor(r(3)/fh));
        pr = Ptx - fspl - wl - slabs * slab(band);
    end
end
