"""
sky.py — Preetham analytical sky model (1999).
Lumion-style HDR sky: sun position + turbidity -> RGB sky colour.

Physics: Rayleigh + Mie scattering (same Mie theory as SEALS notebook §53)
"""
import numpy as np


def preetham_sky(sun_theta: float,
                 view_theta: float,
                 view_gamma: float,
                 turbidity : float = 2.5) -> np.ndarray:
    """
    sun_theta  : sun zenith angle (radians)
    view_theta : view zenith angle (radians)
    view_gamma : angle between view and sun (radians)
    turbidity  : 1=clear, 10=hazy

    Returns xyY (CIE) sky luminance — convert to RGB for rendering.
    """
    # Perez distribution coefficients for Y, x, y
    def F(theta, gamma, A, B, C, D, E):
        return (1 + A*np.exp(B/np.cos(theta))) * \
               (1 + C*np.exp(D*gamma) + E*np.cos(gamma)**2)

    T = turbidity

    # Luminance Y coefficients (Preetham table 2)
    AY= 0.1787*T - 1.4630
    BY=-0.3554*T + 0.4275
    CY=-0.0227*T + 5.3251
    DY= 0.1206*T - 2.5771
    EY=-0.0670*T + 0.3703

    # Chromaticity x coefficients
    Ax=-0.0193*T - 0.2592
    Bx=-0.0665*T + 0.0008
    Cx=-0.0004*T + 0.2125
    Dx=-0.0641*T - 0.8989
    Ex=-0.0033*T + 0.0452

    # Chromaticity y coefficients
    Ay=-0.0167*T - 0.2608
    By=-0.0950*T + 0.0092
    Cy=-0.0079*T + 0.2102
    Dy=-0.0441*T - 1.6537
    Ey=-0.0109*T + 0.0529

    Yz = (4.0453*T - 4.9710) * np.tan((4/9 - T/120)*(np.pi - 2*sun_theta)) \
         - 0.2155*T + 2.4192
    xz = ( 0.00166*T**2*sun_theta**3 - 0.00375*T*sun_theta**3
          + 0.00209*sun_theta**3 + 0.0*T**2*sun_theta**2
          - 0.02903*T*sun_theta**2 + 0.06377*sun_theta**2
          + 0.11693*T*sun_theta - 0.21196*sun_theta + 0.06052*T + 0.25886)
    yz = ( 0.00275*T**2*sun_theta**3 - 0.00610*T*sun_theta**3
          + 0.00317*sun_theta**3 - 0.00421*T**2*sun_theta**2
          + 0.00897*T*sun_theta**2 - 0.04405*sun_theta**2
          + 0.15346*T*sun_theta + 0.26688*sun_theta - 0.26756*T + 0.15643)

    f_ratio_Y = F(view_theta, view_gamma, AY,BY,CY,DY,EY) / \
                F(0,          sun_theta,  AY,BY,CY,DY,EY)
    f_ratio_x = F(view_theta, view_gamma, Ax,Bx,Cx,Dx,Ex) / \
                F(0,          sun_theta,  Ax,Bx,Cx,Dx,Ex)
    f_ratio_y = F(view_theta, view_gamma, Ay,By,Cy,Dy,Ey) / \
                F(0,          sun_theta,  Ay,By,Cy,Dy,Ey)

    Y = Yz * f_ratio_Y
    x = xz * f_ratio_x
    y = yz * f_ratio_y

    # xyY -> XYZ -> linear RGB (sRGB primaries)
    X = (Y/y)*x if y > 0 else 0
    Z = (Y/y)*(1-x-y) if y > 0 else 0

    # sRGB matrix (D65)
    M = np.array([[ 3.2406,-1.5372,-0.4986],
                  [-0.9689, 1.8758, 0.0415],
                  [ 0.0557,-0.2040, 1.0570]])
    rgb = M @ np.array([X, Y, Z]) / 1000.0
    return np.clip(rgb, 0, 1)


def sun_direction(elevation_deg: float, azimuth_deg: float) -> np.ndarray:
    el = np.radians(elevation_deg)
    az = np.radians(azimuth_deg)
    return np.array([np.cos(el)*np.sin(az),
                     np.sin(el),
                     np.cos(el)*np.cos(az)])
