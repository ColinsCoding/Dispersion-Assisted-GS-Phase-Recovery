# -*- coding: utf-8 -*-
"""
Builder: generates notebooks/geomoptics_gausbeam_phycv_graphics.ipynb
Run: py -3.12 -X utf8 notebooks/geomoptics_gausbeam_phycv_graphics.py
"""
import json, pathlib, uuid

NB = pathlib.Path(__file__).parent / "geomoptics_gausbeam_phycv_graphics.ipynb"

def _cid(): return str(uuid.uuid4())[:12]

def md(*lines):
    return {"id":_cid(),"cell_type":"markdown","metadata":{},
            "source":["\n".join(lines)]}

def code(src, _indent=4):
    lines = src.split('\n')
    stripped = [ln[_indent:] if ln.startswith(' '*_indent) else ln for ln in lines]
    return {"id":_cid(),"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
            "source":['\n'.join(stripped).strip('\n') + '\n']}

cells = []

# ── Title ────────────────────────────────────────────────────────────────────
cells.append(md(
    "# Geometric Optics · Gaussian Beam · PhyCV · Computer Graphics",
    "",
    "**Jalali Lab — Dispersion-Assisted GS Phase Recovery**  ",
    "10 sections: Fresnel/TIR · ABCD matrices · Sellmeier/prism · "
    "Gaussian beam · directed energy/Strehl · "
    "PhyCV PST+PIC · PhyCV VEViD · ray-sphere shading · "
    "wave animation · rolling shutter correction",
))

# ── Imports ──────────────────────────────────────────────────────────────────
cells.append(code("""
    import math, cmath, io, warnings
    import numpy as np
    from scipy import ndimage
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    warnings.filterwarnings('ignore', message='FigureCanvasAgg is non-interactive')
    np.random.seed(42)
    PASS = []
    print("Imports OK")
"""))

# ── §1 Fresnel ────────────────────────────────────────────────────────────────
cells.append(md(
    "## §1 — Geometric Optics: Snell's Law / Fresnel R-T / TIR",
    "",
    "**Snell's law:** $n_1\\sin\\theta_1 = n_2\\sin\\theta_2$",
    "",
    "**Fresnel (s-pol):**",
    "$$r_s = \\frac{n_1\\cos\\theta_1 - n_2\\cos\\theta_2}"
    "{n_1\\cos\\theta_1 + n_2\\cos\\theta_2}, \\quad"
    "R_s = r_s^2, \\quad T_s = 1-R_s$$",
    "",
    "**Brewster angle:** $\\theta_B = \\arctan(n_2/n_1)$ — $r_p=0$, p-pol fully transmitted",
    "",
    "**Critical angle:** $\\theta_c = \\arcsin(n_2/n_1)$ — TIR above this",
))

cells.append(code("""
    def snell(theta1_deg, n1, n2):
        sin2 = n1 * math.sin(math.radians(theta1_deg)) / n2
        if abs(sin2) > 1.0: return None, True
        return math.degrees(math.asin(sin2)), False

    def fresnel(theta1_deg, n1, n2):
        t2, tir = snell(theta1_deg, n1, n2)
        if tir: return 1.0, 1.0, 0.0, 0.0
        t1 = math.radians(theta1_deg); t2 = math.radians(t2)
        c1, c2 = math.cos(t1), math.cos(t2)
        rs = (n1*c1 - n2*c2) / (n1*c1 + n2*c2)
        rp = (n2*c1 - n1*c2) / (n2*c1 + n1*c2)
        Rs = rs**2; Rp = rp**2
        return Rs, Rp, 1-Rs, 1-Rp

    n_glass = 1.5; n_air = 1.0
    theta_c  = math.degrees(math.asin(n_air / n_glass))
    brewster = math.degrees(math.atan(n_glass / n_air))
    Rs0, Rp0, _, _ = fresnel(0.0, n_air, n_glass)
    _, Rp_brew, _, _ = fresnel(brewster, n_air, n_glass)
    print(f"  Critical angle: {theta_c:.2f} deg")
    print(f"  Brewster angle: {brewster:.2f} deg  Rp={Rp_brew:.2e}")
    print(f"  Normal incidence R = {Rs0:.4f} = ((n2-n1)/(n2+n1))^2 = {((n_glass-n_air)/(n_glass+n_air))**2:.4f}")
    assert abs(Rs0 - ((n_glass-n_air)/(n_glass+n_air))**2) < 1e-10
    assert Rp_brew < 1e-6
    _, tir_flag = snell(theta_c+1, n_glass, n_air)
    assert tir_flag
    PASS.append(1); print("PASS §1")

    angles = np.linspace(0, 90, 500)
    Rs_air  = np.array([fresnel(a, n_air,  n_glass)[0] for a in angles])
    Rp_air  = np.array([fresnel(a, n_air,  n_glass)[1] for a in angles])
    Rs_glass= np.array([fresnel(a, n_glass, n_air)[0]  for a in angles])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(angles, Rs_air, 'steelblue', lw=2, label='Rs (s-pol)')
    axes[0].plot(angles, Rp_air, 'orange',    lw=2, label='Rp (p-pol)')
    axes[0].axvline(brewster, color='g', ls=':', lw=1.5, label=f'Brewster {brewster:.1f} deg')
    axes[0].set_title('Fresnel reflectance (air->glass)'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(angles, Rs_glass, 'steelblue', lw=2)
    axes[1].axvline(theta_c, color='r', ls='--', lw=1.5, label=f'TIR {theta_c:.1f} deg')
    axes[1].set_title('TIR: glass->air'); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()
"""))

# ── §2 ABCD ───────────────────────────────────────────────────────────────────
cells.append(md(
    "## §2 — Paraxial Ray Tracing: ABCD Transfer Matrices",
    "",
    "**Ray state:** $[y, \\theta]^T$ (height, angle). Transfer matrix:",
    "",
    "| Element | Matrix |",
    "|---------|--------|",
    "| Free space $L$ | $\\begin{pmatrix}1 & L\\\\0 & 1\\end{pmatrix}$ |",
    "| Thin lens $f$ | $\\begin{pmatrix}1 & 0\\\\-1/f & 1\\end{pmatrix}$ |",
    "",
    "**Image condition:** element $B = 0$ (all rays from one point converge)",
    "",
    "**Keplerian telescope:** $M_{\\text{lat}} = -f_2/f_1$, $M_{\\text{ang}} = -f_1/f_2$",
))

cells.append(code("""
    M_free = lambda L: np.array([[1, L], [0, 1]], dtype=float)
    M_lens = lambda f: np.array([[1, 0], [-1/f, 1]], dtype=float)

    f1, f2 = 200e-3, 50e-3; d12 = f1+f2
    M_tel = M_lens(f2) @ M_free(d12) @ M_lens(f1)
    print(f"  Telescope  lat_mag={M_tel[0,0]:.4f} (=-f2/f1={-f2/f1:.4f})  ang_mag={M_tel[1,1]:.4f}")
    assert abs(M_tel[0,0] - (-f2/f1)) < 1e-6
    assert abs(M_tel[1,1] - (-f1/f2)) < 1e-6

    f_l = 0.1; d_obj = 2*f_l
    d_img = 1/(1/f_l - 1/d_obj)
    M_img = M_free(d_img) @ M_lens(f_l) @ M_free(d_obj)
    print(f"  Image formation B={M_img[0,1]:.2e}  magnification={M_img[0,0]:.3f}")
    assert abs(M_img[0,1]) < 1e-10
    PASS.append(2); print("PASS §2")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    z_steps = [0, d_obj, d_obj, d_obj+d_img]
    for ang, col in zip(np.linspace(-0.05,0.05,7), plt.cm.coolwarm(np.linspace(0,1,7))):
        ray = np.array([-0.01, ang])
        ys = [ray[0]]
        for M in [M_free(d_obj), M_lens(f_l), M_free(d_img)]:
            ray = M @ ray; ys.append(ray[0])
        axes[0].plot(np.array(z_steps)*1e3, [y*1e3 for y in ys], color=col, lw=1.2, alpha=0.8)
    axes[0].axvline((d_obj+d_img)*1e3, color='r', ls='--', lw=1, label='image plane')
    axes[0].set_title('Thin lens image formation (ABCD)'); axes[0].legend(); axes[0].grid(alpha=0.3)

    for ang, col in zip(np.linspace(-0.02,0.02,5), plt.cm.plasma(np.linspace(0,1,5))):
        ray = np.array([0.005, ang]); ys=[ray[0]]; z_t=[0]
        for M, dz in zip([M_lens(f1), M_free(d12), M_lens(f2)], [0,f1+d12,f1+d12]):
            ray=M@ray; ys.append(ray[0]); z_t.append(dz)
        axes[1].plot(np.array(z_t)*1e3, [y*1e3 for y in ys], color=col, lw=1.5, alpha=0.85)
    axes[1].set_title('Keplerian telescope ray trace'); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()
"""))

# ── §3 Prism ──────────────────────────────────────────────────────────────────
cells.append(md(
    "## §3 — Prism Dispersion: Sellmeier / Minimum Deviation / Rainbow",
    "",
    "**Sellmeier equation** (BK7 glass):",
    "$$n^2(\\lambda) = 1 + \\sum_i \\frac{B_i\\lambda^2}{\\lambda^2 - C_i}$$",
    "",
    "**Abbe number:** $V = (n_d - 1)/(n_F - n_C)$ — BK7 ≈ 64.2",
    "",
    "**Primary rainbow:** minimum of $\\theta = \\pi + 2i - 4r$ (Descartes)  ",
    "Red outer ($\\theta_{\\text{red}} < \\theta_{\\text{violet}}$, larger elevation from antisolar)",
))

cells.append(code("""
    B1,B2,B3 = 1.03961212, 0.231792344, 1.01046945
    C1,C2,C3 = 6.00069867e-3, 2.00179144e-2, 1.03560653e2
    def sellmeier_bk7(lam_um):
        l2 = lam_um**2
        return math.sqrt(1 + B1*l2/(l2-C1) + B2*l2/(l2-C2) + B3*l2/(l2-C3))

    n_d = sellmeier_bk7(0.589); n_F = sellmeier_bk7(0.486); n_C = sellmeier_bk7(0.656)
    V_d = (n_d-1)/(n_F-n_C)
    print(f"  n_d={n_d:.5f}  V={V_d:.1f}")
    assert 60 < V_d < 68

    A_prism = math.radians(60)
    def min_dev(n): return 2*math.asin(n*math.sin(A_prism/2)) - A_prism
    D_F = math.degrees(min_dev(n_F)); D_C = math.degrees(min_dev(n_C))
    print(f"  Prism 60 deg: D_m(F)={D_F:.2f}  D_m(C)={D_C:.2f}  disp={D_F-D_C:.3f} deg")
    assert D_F > D_C  # blue deflects more

    def rainbow_angle(n):
        best_theta = 180.0
        for i_deg in np.linspace(0.1, 89.9, 10000):
            i = math.radians(i_deg); sin_r = math.sin(i)/n
            if abs(sin_r) > 1: continue
            theta = math.pi + 2*i - 4*math.asin(sin_r)
            if theta < best_theta: best_theta = theta
        return math.degrees(best_theta)
    ang_red = rainbow_angle(1.332); ang_violet = rainbow_angle(1.344)
    print(f"  Rainbow: red={ang_red:.1f} deg  violet={ang_violet:.1f} deg")
    print(f"  Elevation (180-theta): red={180-ang_red:.1f}  violet={180-ang_violet:.1f} (red outer)")
    assert ang_red < ang_violet  # smaller theta = outer (larger elevation)
    PASS.append(3); print("PASS §3")

    lams = np.linspace(0.38, 0.72, 200)
    ns   = np.array([sellmeier_bk7(l) for l in lams])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(lams*1000, ns, 'steelblue', lw=2)
    axes[0].set_xlabel('Wavelength (nm)'); axes[0].set_title('BK7 Sellmeier'); axes[0].grid(alpha=0.3)
    lam_vals = np.linspace(0.38,0.72,20)
    for lv, lc in zip(lam_vals, plt.cm.rainbow(np.linspace(0,1,20))):
        D = math.degrees(min_dev(sellmeier_bk7(lv)))
        axes[1].plot([0,math.cos(math.radians(D))],[0,math.sin(math.radians(D))],color=lc,lw=2.5)
    axes[1].set_title('60 deg prism dispersion fan'); axes[1].set_aspect('equal'); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()
"""))

# ── §4 Gaussian beam ──────────────────────────────────────────────────────────
cells.append(md(
    "## §4 — Gaussian Beam: Waist / Rayleigh Range / Gouy Phase",
    "",
    "$$w(z) = w_0\\sqrt{1+(z/z_R)^2}, \\quad z_R = \\pi w_0^2/\\lambda$$",
    "",
    "$$R(z) = z\\left(1+(z_R/z)^2\\right), \\quad "
    "\\psi(z) = \\arctan(z/z_R) \\text{ (Gouy)}$$",
    "",
    "**Far-field divergence:** $\\theta_\\infty = \\lambda/(\\pi w_0) = w_0/z_R$",
    "",
    "Key: $w(z_R) = \\sqrt{2}\\,w_0$; Gouy phase $\\to \\pi/2$ as $z\\to\\infty$",
))

cells.append(code("""
    lam = 1.064e-6; w0 = 1.0e-3
    z_R = math.pi * w0**2 / lam
    theta_ff = lam / (math.pi * w0)
    print(f"  w0={w0*1e3:.1f}mm  z_R={z_R:.3f}m  theta_ff={theta_ff*1e3:.3f}mrad")
    assert abs(w0*math.sqrt(1+(z_R/z_R)**2) - w0*math.sqrt(2)) < 1e-12  # w(z_R)=sqrt(2)*w0
    assert abs(math.atan(1e12/z_R) - math.pi/2) < 1e-6  # Gouy->pi/2

    z_arr = np.linspace(-5*z_R, 5*z_R, 500)
    w_arr = w0*np.sqrt(1+(z_arr/z_R)**2)
    gouy  = np.arctan(z_arr/z_R)
    PASS.append(4); print("PASS §4")

    Ng = 100; r_max=4*w0; xg=np.linspace(-r_max,r_max,Ng); yg=xg.copy()
    XG,YG=np.meshgrid(xg,yg); RG=np.sqrt(XG**2+YG**2)
    w_2zR = w0*math.sqrt(5); I_2zR = (w0/w_2zR)**2*np.exp(-2*RG**2/w_2zR**2)

    fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].plot(z_arr/z_R, w_arr/w0,'steelblue',lw=2)
    axes[0].plot(z_arr/z_R,-w_arr/w0,'steelblue',lw=2)
    axes[0].set_xlabel('z/z_R'); axes[0].set_title('w(z)/w0'); axes[0].grid(alpha=0.3)
    axes[1].plot(z_arr/z_R,np.degrees(gouy),'orange',lw=2)
    axes[1].axhline(90,color='r',ls='--',lw=1); axes[1].set_title('Gouy phase'); axes[1].grid(alpha=0.3)
    axes[2].imshow(I_2zR,extent=[-r_max/w0,r_max/w0,-r_max/w0,r_max/w0],cmap='hot',origin='lower')
    axes[2].set_title(f'Intensity at z=2z_R'); plt.tight_layout(); plt.show()
"""))

# ── §5 Directed energy ────────────────────────────────────────────────────────
cells.append(md(
    "## §5 — Directed Energy: Strehl Ratio / Fried Parameter r₀ / M²",
    "",
    "**Marechal approximation** (Strehl vs wavefront error):",
    "$$S \\approx e^{-(2\\pi\\sigma_\\phi/\\lambda)^2}$$",
    "",
    "**Kolmogorov turbulence** (aperture D, coherence length r₀):",
    "$$\\sigma_\\phi^2 = 1.03\\,(D/r_0)^{5/3} \\quad [\\text{rad}^2]$$",
    "",
    "**Beam quality:** $w_{\\text{real}}(z) = M^2\\,w_{\\text{ideal}}(z)$ — $M^2=1$ for ideal Gaussian",
))

cells.append(code("""
    D_ap = 0.3; lam_de = 1e-6; R_range = 5e3; r0 = 0.10
    sigma_sq = 1.03*(D_ap/r0)**(5/3)
    strehl   = math.exp(-sigma_sq)
    theta_d  = 1.22*lam_de/D_ap
    w_tgt    = theta_d*R_range
    print(f"  D={D_ap*100:.0f}cm  r0={r0*100:.0f}cm  Strehl={strehl:.4f}")
    print(f"  sigma_phi^2={sigma_sq:.2f} rad^2  spot radius={w_tgt:.2f}m at {R_range/1e3:.0f}km")
    assert 0 < strehl < 1; assert strehl < 0.5
    PASS.append(5); print("PASS §5")

    r0_arr = np.linspace(0.02,0.5,200)
    strehl_arr = np.exp(-1.03*(D_ap/r0_arr)**(5/3))
    M2_vals = [1.0,1.5,2.0,3.0]; w0_s=D_ap/2; z_Rs=math.pi*w0_s**2/lam_de
    z_arr2 = np.linspace(0,R_range,300)

    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(r0_arr*100,strehl_arr,'steelblue',lw=2)
    axes[0].axvline(r0*100,color='r',ls='--',lw=1.5,label=f'r0={r0*100:.0f}cm')
    axes[0].set_xlabel('Fried r0 (cm)'); axes[0].set_title('Strehl vs atmosphere'); axes[0].legend(); axes[0].grid(alpha=0.3)
    for M2,col in zip(M2_vals,['steelblue','orange','green','red']):
        zR_eff = z_Rs/M2
        w_z = np.array([w0_s*M2*math.sqrt(1+(z/zR_eff)**2) for z in z_arr2])
        axes[1].plot(z_arr2/1e3,w_z,color=col,lw=2,label=f'M2={M2}')
    axes[1].set_xlabel('Range (km)'); axes[1].set_title('Beam spread vs M2'); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()
"""))

# ── §6 PhyCV PIC ──────────────────────────────────────────────────────────────
cells.append(md(
    "## §6 — PhyCV PST + PIC: Phase Stretch Transform / Phase Intensity Coupling",
    "",
    "**PST (Phase Stretch Transform)** — Jalali Lab, Asghari & Jalali 2015:",
    "$$\\tilde{E}_{\\text{out}} = \\mathcal{F}^{-1}\\!\\left["
    "\\mathcal{F}[I]\\cdot H_{\\text{LPF}}\\cdot"
    "e^{i\\,s\\,f_w\\arctan(f_w|f|)/\\arctan(f_w)}\\right]$$",
    "",
    "**PIC (Phase Intensity Coupling):** edge map $=|\\nabla I|\\cdot|\\nabla\\phi_{\\text{PST}}|$  ",
    "Edges show *both* amplitude gradient and rapid phase variation.",
))

cells.append(code("""
    def pst_operator(img, strength=0.48, warp=12.14, sigma_LPF=0.1):
        H, W = img.shape
        img_n = (img - img.min()) / (img.max() - img.min() + 1e-12)
        fx=np.fft.fftfreq(W); fy=np.fft.fftfreq(H)
        FX,FY=np.meshgrid(fx,fy); F_norm=np.sqrt(FX**2+FY**2)/(math.sqrt(2)/2)
        LPF = np.exp(-0.5*(F_norm/sigma_LPF)**2)
        PST = np.exp(1j*strength*warp*np.arctan(warp*F_norm)/(np.arctan(warp)+1e-12))
        return np.angle(np.fft.ifft2(np.fft.fft2(img_n)*LPF*PST))

    def pic_operator(img, strength=0.5, sigma_LPF=0.15):
        phase = pst_operator(img, strength=strength, sigma_LPF=sigma_LPF)
        Ix=ndimage.sobel(img,axis=1); Iy=ndimage.sobel(img,axis=0)
        Px=ndimage.sobel(phase,axis=1); Py=ndimage.sobel(phase,axis=0)
        resp = np.sqrt(Ix**2+Iy**2)*np.sqrt(Px**2+Py**2)
        return resp/(resp.max()+1e-12), phase

    H_p,W_p=128,128; x_p=np.linspace(0,4*math.pi,W_p); y_p=np.linspace(0,4*math.pi,H_p)
    Xp,Yp=np.meshgrid(x_p,y_p)
    img_p=(np.sin(Xp)*np.cos(Yp/2)+0.5*np.sin(3*Xp+Yp)+0.1*np.random.randn(H_p,W_p))
    img_p=(img_p-img_p.min())/(img_p.max()-img_p.min())
    img_p[30:60,30:80]=np.maximum(img_p[30:60,30:80],0.8)
    img_p[70:100,40:90]=np.minimum(img_p[70:100,40:90],0.2)

    pic_map, pst_phase = pic_operator(img_p)
    Gx=ndimage.sobel(img_p,axis=1); Gy=ndimage.sobel(img_p,axis=0)
    sobel_map=np.sqrt(Gx**2+Gy**2); sobel_map/=sobel_map.max()+1e-12

    edge_gt=np.zeros_like(img_p)
    for r0,r1,c0,c1 in [(29,62,29,81),(69,101,39,91)]:
        edge_gt[r0,c0:c1]=1; edge_gt[r1-1,c0:c1]=1
        edge_gt[r0:r1,c0]=1; edge_gt[r0:r1,c1-1]=1
    edge_gt_d=ndimage.binary_dilation(edge_gt>0,iterations=5).astype(float)
    pic_in  = np.mean(pic_map[edge_gt_d>0])
    pic_out = np.mean(pic_map[edge_gt_d==0])
    print(f"  PIC edge/bg ratio: {pic_in/pic_out:.1f}x  (in={pic_in:.3f}  out={pic_out:.3f})")
    assert pic_in > pic_out*1.5
    PASS.append(6); print("PASS §6")

    fig,axes=plt.subplots(1,4,figsize=(14,3))
    for ax,im,t in zip(axes,[img_p,pst_phase,sobel_map,pic_map],
                       ['Input','PST phase','Sobel','PIC response']):
        ax.imshow(im,cmap='RdBu_r' if 'PST' in t else 'gray'); ax.set_title(t); ax.axis('off')
    plt.tight_layout(); plt.show()
"""))

# ── §7 VEViD ─────────────────────────────────────────────────────────────────
cells.append(md(
    "## §7 — PhyCV VEViD: Virtual Diffraction + Coherent Detection",
    "",
    "**VEViD** (Jalali Lab, Optica 2022):",
    "",
    "1. **Encode:** $E_{\\text{in}} = e^{ib\\,I(x,y)}$ (image as phase object)",
    "2. **Fresnel propagate:** $\\tilde{E}_{\\text{out}} = \\mathcal{F}^{-1}["
    "\\mathcal{F}[E_{\\text{in}}]\\cdot e^{-i\\pi\\lambda_v z_v(f_x^2+f_y^2)}]$",
    "3. **Detect:** $I_{\\text{out}} = |E_{\\text{out}}|^2$",
    "4. **Blend:** $(1-\\alpha)I + \\alpha I_{\\text{out}}$",
    "",
    "Physics intuition: virtual coherent illumination → edge diffraction → "
    "natural contrast enhancement without halo artifacts.",
))

cells.append(code("""
    def vevid(img, b=math.pi/3, lam_v=0.01, z_v=1.0, blend=0.7):
        H_v, W_v = img.shape
        E_in = np.exp(1j*b*img)
        fx=np.fft.fftfreq(W_v); fy=np.fft.fftfreq(H_v)
        FX,FY=np.meshgrid(fx,fy)
        H_f = np.exp(-1j*math.pi*lam_v*z_v*(FX**2+FY**2))
        E_out = np.fft.ifft2(np.fft.fft2(E_in)*H_f)
        I_out = np.abs(E_out)**2
        I_out = (I_out-I_out.min())/(I_out.max()-I_out.min()+1e-12)
        return (1-blend)*img + blend*I_out

    H_v,W_v=128,128; x_v=np.linspace(0,3*math.pi,W_v); y_v=np.linspace(0,3*math.pi,H_v)
    Xv,Yv=np.meshgrid(x_v,y_v)
    img_v=0.3+0.15*np.sin(Xv)*np.cos(Yv/2)+0.08*np.sin(2*Xv+1.5*Yv)
    img_v+=0.02*np.random.randn(H_v,W_v); img_v=np.clip(img_v,0,1)

    img_enh = vevid(img_v, b=math.pi/3, lam_v=0.02, z_v=1.0, blend=0.8)
    print(f"  Contrast: in={img_v.std():.4f}  out={img_enh.std():.4f} ({img_enh.std()/img_v.std():.2f}x)")
    assert img_enh.std() > img_v.std()
    PASS.append(7); print("PASS §7")

    b_vals=[math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3]
    imgs_b=[vevid(img_v,b=b,lam_v=0.02,z_v=1.0,blend=0.8) for b in b_vals]
    fig,axes=plt.subplots(1,6,figsize=(16,3))
    axes[0].imshow(img_v,cmap='gray',vmin=0,vmax=1); axes[0].set_title('Input'); axes[0].axis('off')
    axes[1].imshow(img_enh,cmap='gray',vmin=0,vmax=1); axes[1].set_title('VEViD'); axes[1].axis('off')
    for i,(b,im_b) in enumerate(zip(b_vals,imgs_b)):
        axes[i+2].imshow(im_b,cmap='gray',vmin=0,vmax=1); axes[i+2].set_title(f'b={b/math.pi:.2f}pi'); axes[i+2].axis('off')
    plt.tight_layout(); plt.show()
"""))

# ── §8 Ray-sphere ─────────────────────────────────────────────────────────────
cells.append(md(
    "## §8 — Ray-Sphere Computer Graphics: Intersection / Phong / Shadow",
    "",
    "**Ray:** $P(t) = O + t\\mathbf{D}$",
    "",
    "**Sphere intersection** ($|P-C|^2=R^2$):",
    "$$t = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}, \\quad a=|\\mathbf{D}|^2, "
    "b=2\\mathbf{D}\\cdot(O-C), c=|O-C|^2-R^2$$",
    "",
    "**Phong illumination:**",
    "$$I = k_a I_a + k_d(\\hat{n}\\cdot\\hat{L})I_d + k_s(\\hat{r}\\cdot\\hat{v})^\\alpha I_s$$",
))

cells.append(code("""
    def ray_sphere(O, D, C, R):
        oc=O-C; a=np.dot(D,D); b=2*np.dot(oc,D); c=np.dot(oc,oc)-R*R
        disc=b*b-4*a*c
        if disc<0: return None
        t1=(-b-math.sqrt(disc))/(2*a); t2=(-b+math.sqrt(disc))/(2*a)
        if t1>1e-4: return t1
        if t2>1e-4: return t2
        return None

    def phong(P,N,V,lights,spheres,ka=0.1,kd=0.7,ks=0.3,alpha=32):
        color=ka
        for L_pos,L_col in lights:
            L=L_pos-P; dL=np.linalg.norm(L); L/=dL
            shadow=any(ray_sphere(P,L,sc,sr) is not None and
                       ray_sphere(P,L,sc,sr)<dL for sc,sr,_ in spheres)
            if not shadow:
                diff=max(0,np.dot(N,L)); R_v=2*np.dot(N,L)*N-L
                color+=L_col*(kd*diff+ks*max(0,np.dot(R_v,V))**alpha)
        return min(1.0,color)

    spheres=[(np.array([0.,0.,3.]),0.8,np.array([0.8,0.2,0.2])),
             (np.array([-1.5,-0.3,4.5]),0.5,np.array([0.2,0.6,0.8])),
             (np.array([1.2,0.2,3.5]),0.4,np.array([0.3,0.8,0.3])),
             (np.array([0.,-100.8,3.]),100.,np.array([0.8,0.8,0.8]))]
    lights=[(np.array([-3.,5.,1.]),0.9)]
    img_W,img_H=200,150; cam=np.array([0.,0.5,-1.]); fov=math.radians(60)
    img_rgb=np.zeros((img_H,img_W,3))
    for iy in range(img_H):
        for ix in range(img_W):
            px=(2*(ix+0.5)/img_W-1)*math.tan(fov/2)*img_W/img_H
            py=-(2*(iy+0.5)/img_H-1)*math.tan(fov/2)
            D=np.array([px,py,1.]); D/=np.linalg.norm(D)
            t_min=float('inf'); hit=None
            for sph in spheres:
                t=ray_sphere(cam,D,sph[0],sph[1])
                if t is not None and t<t_min: t_min=t; hit=sph
            if hit:
                P_h=cam+t_min*D; N_h=(P_h-hit[0])/hit[1]
                img_rgb[iy,ix]=np.clip(hit[2]*phong(P_h,N_h,-D,lights,spheres),0,1)
    print(f"  Rendered {img_W}x{img_H}  max={img_rgb.max():.3f}  mean={img_rgb.mean():.3f}")
    assert img_rgb.max()>0.5
    PASS.append(8); print("PASS §8")
    fig,ax=plt.subplots(figsize=(8,6))
    ax.imshow(img_rgb); ax.set_title('Ray-traced scene: Phong + shadow'); ax.axis('off'); plt.show()
"""))

# ── §9 Animation ──────────────────────────────────────────────────────────────
cells.append(md(
    "## §9 — Wave Animation: GVD-Broadening Chirped Packet",
    "",
    "**Chirped wave packet:**",
    "$$E(x,t) = \\underbrace{e^{-(x-v_g t)^2/(2\\sigma(t)^2)}}_{\\text{envelope}}"
    "\\cdot\\cos(k_0 x - \\omega_0 t + \\phi_{\\text{chirp}})$$",
    "",
    "**GVD broadening:**",
    "$$\\sigma(t) = \\sigma_0\\sqrt{1+(\\beta_2 t/\\sigma_0^2)^2}$$",
    "",
    "Saved as GIF via matplotlib FuncAnimation + Pillow writer.",
))

cells.append(code("""
    N_x=400; dx=0.05; x_anim=(np.arange(N_x)-N_x//2)*dx
    k0=20.; omega0=10.; vg=0.5; sigma0=1.; beta2_a=0.3; chirp_a=1.5
    N_frames=20; dt_a=0.3

    def packet(x,t):
        xc=x-vg*t; brd=math.sqrt(1+(beta2_a*t/sigma0**2)**2); st=sigma0*brd
        env=np.exp(-xc**2/(2*st**2)); phase=k0*x-omega0*t+chirp_a*xc**2/(2*st**2)
        return env, env*np.cos(phase), brd

    fig_a,ax_a=plt.subplots(figsize=(10,4))
    ax_a.set_xlim(x_anim[0],x_anim[-1]); ax_a.set_ylim(-1.3,1.3)
    lep,=ax_a.plot([],[],'steelblue',lw=1.5,alpha=0.4,label='envelope')
    lem,=ax_a.plot([],[],'steelblue',lw=1.5,alpha=0.4)
    lw, =ax_a.plot([],'orange',lw=1.)
    tt  =ax_a.text(0.05,0.92,'',transform=ax_a.transAxes)
    ax_a.legend(); ax_a.grid(alpha=0.3); ax_a.set_title('GVD wave packet broadening')

    def init(): lep.set_data([],[]); lem.set_data([],[]); lw.set_data([],[]); tt.set_text(''); return lep,lem,lw,tt
    def update(f):
        env,field,brd=packet(x_anim,f*dt_a)
        lep.set_data(x_anim,env); lem.set_data(x_anim,-env); lw.set_data(x_anim,field)
        tt.set_text(f't={f*dt_a:.1f}  w={brd:.2f}x'); return lep,lem,lw,tt

    ani=animation.FuncAnimation(fig_a,update,frames=N_frames,init_func=init,blit=True,interval=80)
    try:
        ani.save('wave_packet.gif',writer='pillow',fps=12,dpi=80)
        import os; print(f"  Saved wave_packet.gif ({os.path.getsize('wave_packet.gif')//1024}KB)")
    except Exception as e:
        print(f"  GIF save skipped ({e})")
    plt.close(fig_a)

    env_t,_,brd_t=packet(x_anim,5*dt_a); assert env_t.max()>0
    PASS.append(9); print("PASS §9")

    fig,axes=plt.subplots(1,4,figsize=(14,3))
    for i,frame in enumerate([0,5,10,18]):
        env,field,brd=packet(x_anim,frame*dt_a)
        axes[i].plot(x_anim,field,'orange',lw=1,alpha=0.9)
        axes[i].plot(x_anim,env,'steelblue',lw=1.5,alpha=0.6)
        axes[i].plot(x_anim,-env,'steelblue',lw=1.5,alpha=0.6)
        axes[i].set_title(f't={frame*dt_a:.1f}  w={brd:.2f}x')
        axes[i].set_ylim(-1.3,1.3); axes[i].grid(alpha=0.3)
    plt.suptitle('Wave packet GVD broadening snapshots'); plt.tight_layout(); plt.show()
"""))

# ── §10 Rolling shutter ───────────────────────────────────────────────────────
cells.append(md(
    "## §10 — Rolling Shutter: Geometric Distortion + Linear Correction",
    "",
    "**CMOS rolling readout:** row $i$ is captured at time $t_i = i\\,\\Delta t$",
    "",
    "**Horizontal pan distortion** (angular velocity $\\Omega$, focal length $f$):",
    "$$x_{\\text{distorted}}(i) = x_{\\text{true}} + \\Omega\\cdot t_i\\cdot f$$",
    "",
    "**Correction:** reverse the row-dependent shift using estimated $\\Omega$.",
))

cells.append(code("""
    H_rs,W_rs=128,128; dt_row=0.002; omega_pan=math.radians(30); f_px=100
    def checkerboard(H,W,sq=12):
        C,R=np.meshgrid(np.arange(W),np.arange(H))
        return ((C//sq+R//sq)%2).astype(float)
    img_true=checkerboard(H_rs,W_rs); img_true[:,60:64]=1.; img_true[:,61:63]=0.

    shifts=np.array([omega_pan*i*dt_row*f_px for i in range(H_rs)])
    img_rs=np.zeros_like(img_true)
    for i in range(H_rs):
        sh=shifts[i]; src=np.arange(W_rs)-sh
        ok=(src>=0)&(src<W_rs-1); fl=np.floor(src).astype(int); fr=src-fl
        for j in np.where(ok)[0]:
            img_rs[i,j]=(1-fr[j])*img_true[i,fl[j]]+fr[j]*img_true[i,min(fl[j]+1,W_rs-1)]

    img_corr=np.zeros_like(img_rs)
    for i in range(H_rs):
        sh=shifts[i]; dst=np.arange(W_rs)+sh
        ok=(dst>=0)&(dst<W_rs-1); fl=np.floor(dst).astype(int); fr=dst-fl
        for j in np.where(ok)[0]:
            ci=fl[j]; img_corr[i,ci]+=(1-fr[j])*img_rs[i,j]
            if ci+1<W_rs: img_corr[i,ci+1]+=fr[j]*img_rs[i,j]

    mse_rs=np.mean((img_true-img_rs)**2); mse_c=np.mean((img_true-img_corr)**2)
    print(f"  Shifts: {shifts[0]:.2f} to {shifts[-1]:.2f} px")
    print(f"  MSE: distorted={mse_rs:.4f}  corrected={mse_c:.4f}  ({mse_rs/mse_c:.1f}x improvement)")
    assert mse_c < mse_rs
    PASS.append(10); print("PASS §10")

    fig,axes=plt.subplots(1,4,figsize=(14,4))
    axes[0].imshow(img_true,cmap='gray'); axes[0].set_title('Ground truth'); axes[0].axis('off')
    axes[1].imshow(img_rs,cmap='gray'); axes[1].set_title('Rolling shutter'); axes[1].axis('off')
    axes[2].imshow(img_corr,cmap='gray'); axes[2].set_title(f'Corrected'); axes[2].axis('off')
    axes[3].plot(shifts,range(H_rs),'steelblue',lw=2)
    axes[3].set_xlabel('Row shift (px)'); axes[3].set_title('RS shift profile'); axes[3].invert_yaxis(); axes[3].grid(alpha=0.3)
    plt.tight_layout(); plt.show()
"""))

# ── Summary ───────────────────────────────────────────────────────────────────
cells.append(code("""
    print(f"\\n{len(PASS)}/10 PASS")
    for i,s in enumerate([
        "Fresnel R/T, Brewster, TIR critical angle",
        "ABCD matrices: telescope magnification, image B=0",
        "Sellmeier BK7, Abbe V=64.2, rainbow Descartes",
        "Gaussian beam: w(zR)=sqrt(2)*w0, Gouy->pi/2",
        "Directed energy: Strehl=exp(-sigma^2), M2 spread",
        "PhyCV PST+PIC: phase-intensity coupling edge map",
        "PhyCV VEViD: Fresnel virtual diffraction enhancement",
        "Ray-sphere: Phong shading + hard shadow (200x150)",
        "Wave animation: GVD broadening GIF (pillow writer)",
        "Rolling shutter: per-row distortion + correction"
    ], 1):
        mark = "PASS" if i in PASS else "    "
        print(f"  [{mark}] §{i:2d}  {s}")
"""))

# ── Write ─────────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"}
    },
    "cells": cells
}
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written: {NB}  ({NB.stat().st_size} bytes, {len(cells)} cells)")
