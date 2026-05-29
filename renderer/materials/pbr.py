"""
pbr.py — Physically Based Rendering material.
Cook-Torrance BRDF:  f = (D*F*G) / (4*cos_i*cos_o)

Physics:
  D = GGX normal distribution   (microfacet distribution)
  F = Schlick Fresnel           (reflectance vs angle)
  G = Smith geometry term       (self-shadowing)

Griffiths EM ch9 connection:
  Fresnel equations at interface:
  F(0) = ((n1-n2)/(n1+n2))^2
  Schlick approx: F(theta) = F0 + (1-F0)*(1-cos(theta))^5
"""
import numpy as np


def schlick_fresnel(cos_theta: float, F0: float) -> float:
    return F0 + (1 - F0) * (1 - cos_theta) ** 5


def ggx_ndf(NdotH: float, roughness: float) -> float:
    """GGX / Trowbridge-Reitz normal distribution."""
    a  = roughness ** 2
    a2 = a ** 2
    d  = (NdotH**2 * (a2 - 1) + 1)
    return a2 / (np.pi * d**2 + 1e-9)


def smith_geometry(NdotV: float, NdotL: float, roughness: float) -> float:
    k  = ((roughness + 1)**2) / 8
    gv = NdotV / (NdotV*(1-k) + k + 1e-9)
    gl = NdotL / (NdotL*(1-k) + k + 1e-9)
    return gv * gl


def cook_torrance(NdotL, NdotV, NdotH, HdotV,
                  albedo, metallic, roughness):
    """
    Full PBR shading — numpy arrays or scalars.
    Returns RGB radiance.
    """
    albedo   = np.array(albedo, dtype=float)
    F0       = 0.04 * (1 - metallic) + albedo * metallic   # dielectric/metal blend

    D = ggx_ndf(NdotH, roughness)
    F = schlick_fresnel(HdotV, F0.mean())
    G = smith_geometry(NdotV, NdotL, roughness)

    specular = (D * F * G) / (4 * NdotL * NdotV + 1e-9)
    diffuse  = (1 - F) * (1 - metallic) * albedo / np.pi

    return (diffuse + specular) * NdotL


class PBRMaterial:
    def __init__(self,
                 albedo   = (0.8, 0.6, 0.2),   # golden doodle default
                 metallic  = 0.0,
                 roughness = 0.6,
                 emission  = (0, 0, 0)):
        self.albedo    = np.array(albedo)
        self.metallic  = metallic
        self.roughness = roughness
        self.emission  = np.array(emission)

    def shade(self, normal, view_dir, light_dir, light_color=(1,1,1)):
        N = normal / (np.linalg.norm(normal) + 1e-9)
        V = view_dir / (np.linalg.norm(view_dir) + 1e-9)
        L = light_dir / (np.linalg.norm(light_dir) + 1e-9)
        H = (V + L) / (np.linalg.norm(V + L) + 1e-9)

        NdotL = max(np.dot(N, L), 0)
        NdotV = max(np.dot(N, V), 1e-4)
        NdotH = max(np.dot(N, H), 0)
        HdotV = max(np.dot(H, V), 0)

        radiance = cook_torrance(NdotL, NdotV, NdotH, HdotV,
                                 self.albedo, self.metallic, self.roughness)
        return radiance * np.array(light_color) + self.emission
