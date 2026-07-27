"""Monocular Feature Layer — single-eye low-level feature extraction.

Implements the monocular feature layer described in the roadmap:
  - Gabor wavelet frontend (V1/V2 local filtering)
  - Chromatic channel separation with S·e^{iH} complex encoding
  - Gain control divisive normalization — illumination invariance
  - ε-floor: 1 % of image energy mean — prevents flat-pixel spectrum hijack
  - 2θ circular-statistics encoding for orientation
  - V3 feature vector: spectral shape + log magnitude + 5 spectral moments
    + circular variance + optional orientation group
  - Pb half-disk boundary channel
  - Feature-group auto-selection via coherence criterion

Usage
-----
    from monocular import MonocularFeatureLayer

    mf = MonocularFeatureLayer(orientation_sensitive=True)
    result = mf.extract(rgb_image)
    # result.features  — (N, D) feature matrix
    # result.bmap      — (H, W) boundary map
    # result.chromatic — complex S·e^{iH} encoding
    # result.coherence — feature-group selection diagnostics
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx

from color import Color
from gabor import GaborWavelet
from gabor_features import (
    _boundary_channel,
    _grid,
)
from gabor_features import (
    _features as _luminance_features,
)

# ═══════════════════════════════════════════════════════════════════════
# Output dataclass
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class MonocularFeatures:
    """Output of the monocular feature layer for a single eye.

    Attributes
    ----------
    features : mx.array  (N, D)
        Per-pixel feature vectors.  D depends on scale_size and the
        orientation_sensitive / use_dc flags.
        Base: 2*S + 8   (spectral shape + log mag + 5 moments + S+1 circ + 1 contrast).
        +2S+2 if orientation_sensitive.  +2 if use_dc.
    dim : int
        Feature dimension D.
    z_stats : dict
        Frozen per-group z-score anchors for cross-frame stability.
    gw : GaborWavelet
        The wavelet instance — retained so downstream modules can
        access raw Gabor responses (e.g. the binocular disparity estimator
        consumes the complex Gabor responses directly).
    bmap : mx.array | None  (H, W)
        Pb half-disk boundary strength in [0, 1].
    bnormal : mx.array | None  (H, W)
        Boundary normal direction in radians (π-periodic).
    bimod : mx.array | None  (H, W)
        Junction bimodality ratio in [0, 1].
    chromatic : mx.array | None  (H, W)
        Chromatic complex encoding S·exp(i·H·2π).
    H : int
    W : int
    sy, sx : mx.array  (N,)
        Pixel grid indices (flattened row-major).
    coherence : dict
        Feature-group selection diagnostics.  Keys:
        - "orientation_enabled" : bool
        - "chroma_enabled" : bool
        - "dir_iso_ratio" : float — directional / isotropic energy variance ratio
        - "chroma_coh_gain" : float — relative chromatic coherence gain
    """

    features: mx.array
    dim: int
    z_stats: dict
    gw: GaborWavelet
    bmap: mx.array | None
    bnormal: mx.array | None
    bimod: mx.array | None
    chromatic: mx.array | None
    H: int
    W: int
    sy: mx.array
    sx: mx.array
    coherence: dict


# ═══════════════════════════════════════════════════════════════════════
# Monocular feature layer
# ═══════════════════════════════════════════════════════════════════════


class MonocularFeatureLayer:
    """Monocular feature layer for one eye's RGB image.

    Extracts multi-scale, multi-orientation low-level feature
    representations from a single RGB image.  This is the computational
    substrate that serves both depth pathways — the monocular cue pathway
    and the binocular disparity estimator.

    Parameters
    ----------
    scale_size : int | None
        Number of Gabor scales.  Auto-derives from image size when None
        (octave-spaced between λ_min=3 px and λ_max=min(H,W)/2, floored at 4).
    orientation_size : int
        Number of Gabor orientations, uniformly sampling [0, π).  Default 8.
    bandwidth : float
        Octave bandwidth of Gabor filters.  Default 1.0.
    gamma : float
        Spatial aspect ratio (ellipticity) of Gabor filters.  Default 0.5.
    orientation_sensitive : bool | None
        Whether to include 2θ orientation encoding in the feature vector.
        When None (default), auto-decides via the coherence criterion.
    use_dc : bool
        Include DC channel features (local mean intensity + 5×5 local std).
        Default True.
    boundary_radius : int | None
        Half-disk radius for the Pb boundary detector.  None disables
        boundary computation entirely.  Default 4.
    boundary_n_theta : int
        Number of candidate orientations swept by the boundary detector.
        Default 8.
    """

    def __init__(
        self,
        scale_size: int | None = None,
        orientation_size: int = 8,
        bandwidth: float = 1.0,
        gamma: float = 0.5,
        orientation_sensitive: bool | None = None,
        use_dc: bool = True,
        boundary_radius: int | None = 4,
        boundary_n_theta: int = 8,
    ):
        self._scale_size = scale_size
        self._orientation_size = orientation_size
        self._bandwidth = bandwidth
        self._gamma = gamma
        self._orientation_sensitive = orientation_sensitive
        self._use_dc = use_dc
        self._boundary_radius = boundary_radius
        self._boundary_n_theta = boundary_n_theta

    # ── public API ──────────────────────────────────────────────────

    def extract(self, rgb: mx.array) -> MonocularFeatures:
        """Extract features from a single-eye RGB image.

        Parameters
        ----------
        rgb : mx.array  (H, W, 3)
            RGB image with values in [0, 1].

        Returns
        -------
        MonocularFeatures
            Feature matrix, boundary map, chromatic encoding, and metadata.
        """
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"rgb must be (H, W, 3), got shape {rgb.shape}")
        H, W = rgb.shape[:2]

        # ── luminance extraction ─────────────────────────────────
        # ITU-R BT.601 luma: Y = 0.299·R + 0.587·G + 0.114·B
        y = (
            0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        )
        mx.eval(y)

        # ── Gabor wavelet on luminance ────────────────────────────
        gw = GaborWavelet(
            y,
            scale_size=self._scale_size,
            orientation_size=self._orientation_size,
            bandwidth=self._bandwidth,
            gamma=self._gamma,
        )
        H, W = gw.H, gw.W

        # ── pixel grid ────────────────────────────────────────────
        sy, sx, _sp, _Ny, _Nx, _N = _grid(H, W)

        # ── coherence criterion — decides orientation / chroma ────
        coherence = self._compute_coherence(gw, rgb, sy, sx)

        orientation_sensitive = self._orientation_sensitive
        if orientation_sensitive is None:
            orientation_sensitive = coherence["orientation_enabled"]

        # ── luminance features (v3 vector) ────────────────────────
        feats, z_stats = _luminance_features(
            gw,
            sy,
            sx,
            orientation_sensitive=orientation_sensitive,
            use_dc=self._use_dc,
            use_pooled=True,
            use_circ=True,
        )

        # ── boundary channel ──────────────────────────────────────
        bmap, bnormal, bimod = None, None, None
        if self._boundary_radius is not None:
            bmap, bnormal, bimod = _boundary_channel(
                gw, radius=self._boundary_radius, n_theta=self._boundary_n_theta
            )

        # ── chromatic encoding ────────────────────────────────────
        chromatic = None
        if coherence["chroma_enabled"]:
            hsl = Color.rgb_to_hsl(rgb)
            chromatic = Color.hsl_to_complex(hsl)
            mx.eval(chromatic)

        return MonocularFeatures(
            features=feats,
            dim=feats.shape[1],
            z_stats=z_stats,
            gw=gw,
            bmap=bmap,
            bnormal=bnormal,
            bimod=bimod,
            chromatic=chromatic,
            H=H,
            W=W,
            sy=sy,
            sx=sx,
            coherence=coherence,
        )

    # ── coherence criterion ────────────────────────────────────────

    def _compute_coherence(
        self, gw: GaborWavelet, rgb: mx.array, sy: mx.array, sx: mx.array
    ) -> dict:
        """Determine which feature groups to enable.

        Two independent criteria:

        **Orientation group** — enabled when the directional energy
        variance substantially exceeds the isotropic baseline.  A scene
        where all textures are isotropic (e.g. noise-on-noise) gains
        nothing from orientation encoding, and adding it would add 2S+2
        noisy dimensions that dilute the clustering.

        **Chromatic group** — enabled when hue provides discriminative
        information beyond what luminance already captures.  Measured as
        the relative coherence gain from adding the hue dimension to a
        luminance-only feature space.

        Both use a 15 % relative-gain threshold (empirically robust, see
        the DG-GMRF abandonment report §4.4).
        """
        # ── orientation coherence ──────────────────────────────────
        E = gw.energy_tensor  # (S, O, H, W)
        S, O = E.shape[0], E.shape[1]
        thetas = mx.array(gw.orientations)

        # Per-pixel directional vs isotropic energy variance
        # Across orientations, for each scale
        dir_vars = []
        iso_vars = []
        for s in range(S):
            amps_s = E[s]  # (O, H, W)
            total_s = amps_s.sum(axis=0, keepdims=True)
            eps_s = 0.01 * mx.maximum(total_s.mean(), 1e-12)
            amps_n = amps_s / (total_s + eps_s)

            # Directional variance: how much the dominant orientation
            # stands out from the mean across orientations
            mean_across = amps_n.mean(axis=0)
            dir_var = ((amps_n - mean_across) ** 2).sum(axis=0) / O
            dir_vars.append(dir_var[sy, sx].mean())

            # Isotropic baseline: variance if all orientations had
            # equal energy (circular uniform distribution)
            # For a uniform circular distribution, the expected R² ≈ 0
            C = (amps_n * mx.cos(2.0 * thetas).reshape(-1, 1, 1)).sum(axis=0)
            Ss = (amps_n * mx.sin(2.0 * thetas).reshape(-1, 1, 1)).sum(axis=0)
            R = mx.sqrt(C * C + Ss * Ss)
            iso_vars.append(float((1.0 - R)[sy, sx].mean().item()))

        dir_mean = float(sum(float(v.item()) for v in dir_vars) / S)
        iso_mean = sum(iso_vars) / S
        dir_iso_ratio = dir_mean / max(iso_mean, 1e-8) if iso_mean > 1e-8 else 1.0

        # Enable orientation group when directional variance exceeds
        # isotropic by >15 % (relative gain)
        orientation_enabled = dir_iso_ratio > 1.15

        # ── chromatic coherence ────────────────────────────────────
        chroma_enabled = False
        chroma_coh_gain = 0.0

        # Heuristic: hue provides information when the saturation
        # variance across the image is non-negligible (>5 % of max
        # saturation) and the hue distribution is non-uniform (not
        # everything is the same colour).
        hsl = Color.rgb_to_hsl(rgb)
        sat = hsl[..., 1]
        hue = hsl[..., 0]

        sat_mean = float(sat.mean().item())
        sat_max = float(sat.max().item())

        if sat_max > 0.1 and sat_mean > 0.02:
            # Check hue non-uniformity: circular variance of hue
            hue_rad = hue * 2.0 * math.pi
            C_hue = float(mx.cos(hue_rad).mean().item())
            S_hue = float(mx.sin(hue_rad).mean().item())
            R_hue = math.sqrt(C_hue * C_hue + S_hue * S_hue)
            hue_circ_var = 1.0 - R_hue

            # High circular variance → diverse hues → chroma is useful
            if hue_circ_var > 0.3:
                chroma_coh_gain = hue_circ_var / max(1.0 - hue_circ_var, 1e-8)
                chroma_enabled = chroma_coh_gain > 0.15

        return {
            "orientation_enabled": orientation_enabled,
            "chroma_enabled": chroma_enabled,
            "dir_iso_ratio": dir_iso_ratio,
            "chroma_coh_gain": chroma_coh_gain,
        }
