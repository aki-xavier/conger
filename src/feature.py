# """Gabor feature pipeline for pixel-wise clustering.

# Per-pixel features (each group z-normalised unless noted):
#   spectral shape (S, gain-controlled) + log magnitude (1)
#   + pooled spectral moments (5) + circular variance (S+1)
#   + local contrast (1, RMS-normalised) = 2S + 8 base dims
#   + orientation components (2S+2, also z-normalised) if orientation_sensitive
#   + DC channel (2) if use_dc

# Magnitude features are divisively normalised per pixel (V1-style gain
# control): a smooth multiplicative illumination field scales every Gabor
# channel equally, so spectral *shape* and RMS-relative contrast are
# illumination-invariant — shading cannot hijack the clustering.

# Also provides the pixel-grid / neighbour-graph construction (_grid, _pairs)
# and the boundary channel (_boundary_channel): Pb-style half-disk texture
# contrast, supplying edge strength and boundary normal for the pairwise
# weights and the co-circularity (association-field) kernel.
# """

# import mlx.core as mx


# def _z(
#     feats: mx.array, stats: tuple[mx.array, mx.array] | None = None
# ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
#     """z-normalise a feature group. stats=(mean, std) freezes the anchors
#     for cross-frame stability (temporal use: the same world point must
#     keep the same feature coordinates across frames). Returns (z, stats).
#     """
#     if stats is None:
#         m = feats.mean(axis=0, keepdims=True)
#         s = mx.maximum(feats.std(axis=0, keepdims=True), 1e-8)
#         stats = (m, s)
#     m, s = stats
#     return (feats - m) / s, stats


# def _scale_energy(gw, sy: mx.array, sx: mx.array, stats=None) -> tuple[mx.array, tuple]:
#     """Spectral shape of the Gabor response: per-scale amplitude pooled
#     across all orientations, divisively normalised per pixel by the total
#     energy (V1-style gain control, Carandini & Heeger 2012). A smooth
#     multiplicative illumination field scales every channel equally, so
#     the shape is illumination-invariant; flat pixels map to the origin
#     via the eps floor. A single log-compressed magnitude channel (energy
#     relative to the image mean) keeps flat-vs-textured discriminable
#     without re-opening the door to shading hijack. Returns (feats, stats)
#     with stats = (z_mean, z_std, total_energy_mean) — pass it back to
#     freeze all per-image anchors (temporal use).
#     """
#     S = gw.scale_size
#     E = gw.energy_tensor  # (S, O, H, W)
#     maps = [E[s].sum(axis=0) for s in range(S)]
#     total = maps[0]
#     for m in maps[1:]:
#         total = total + m
#     if stats is None:
#         tmean = mx.maximum(total.mean(), 1e-12)
#     else:
#         _zm, _zs, tmean = stats
#     eps = 0.01 * tmean
#     denom = total + eps
#     cols = [(m / denom)[sy, sx].reshape(-1, 1) for m in maps]
#     # one compressed magnitude channel: log energy relative to the image
#     # mean. Keeps "flat vs textured" (and strong contrast) discriminable
#     # — without it, flat+noise is confidently confused with a true
#     # noise texture — while the log compression bounds the influence of
#     # a smooth illumination field (100× energy range → 4.6 nats on one
#     # dim).
#     mag = mx.log(denom / (tmean + eps))
#     cols.append(mag[sy, sx].reshape(-1, 1))
#     z, zs = _z(mx.concatenate(cols, axis=1), None if stats is None else stats[:2])
#     return z, (zs[0], zs[1], tmean)


# def _pooled_spectral(
#     gw, sy: mx.array, sx: mx.array, stats=None
# ) -> tuple[mx.array, tuple]:
#     """Spectral moments on orientation-pooled scale energy.
#     Returns (feats, stats): (N, 5) centroid, variance, skewness, kurtosis,
#     rolloff. Pass stats back to freeze the z-norm anchors (temporal use).

#     All moments are computed across the scale (frequency) axis, ordered
#     from fine (small λ, high f) to coarse (large λ, low f).  Rolloff is
#     the frequency above which 85 % of the cumulative energy sits (low-pass
#     direction: accumulate from coarse toward fine), i.e. 85 % of energy
#     is concentrated at frequencies ≤ rolloff.  This is the inverse of the
#     more common high-pass rolloff; the low-pass convention was chosen
#     because the coarsest scales carry the structural signal in texture
#     segmentation.
#     """
#     S = gw.scale_size
#     H, W = gw.H, gw.W
#     E = gw.energy_tensor  # (S, O, H, W)

#     scale_E = [E[s].sum(axis=0) for s in range(S)]  # per-scale pooled → (H, W)
#     total = mx.zeros((H, W), dtype=mx.float32)
#     for e in scale_E:
#         total = total + e
#     safe_total = mx.maximum(total, 1e-12)

#     f = mx.array([1.0 / lam for lam in gw.wavelengths])
#     centroid = mx.zeros((H, W), dtype=mx.float32)
#     for fi, e_s in zip(f, scale_E, strict=True):
#         centroid = centroid + fi * e_s / safe_total

#     variance = mx.zeros((H, W), dtype=mx.float32)
#     skewness = mx.zeros((H, W), dtype=mx.float32)
#     kurtosis = mx.zeros((H, W), dtype=mx.float32)
#     for fi, e_s in zip(f, scale_E, strict=True):
#         p = e_s / safe_total
#         diff = fi - centroid
#         variance = variance + diff * diff * p
#         skewness = skewness + diff * diff * diff * p
#         kurtosis = kurtosis + diff * diff * diff * diff * p

#     sigma = mx.sqrt(mx.maximum(variance, 1e-12))
#     skewness = skewness / mx.maximum(sigma**3, 1e-12)
#     kurtosis = kurtosis / mx.maximum(sigma**4, 1e-12)

#     # low-pass rolloff: accumulate from coarsest (largest λ) toward finest;
#     # the returned value is the frequency above which 85 % of energy sits.
#     cum = mx.zeros((H, W), dtype=mx.float32)
#     rolloff = mx.zeros((H, W), dtype=mx.float32)
#     remaining = mx.ones((H, W), dtype=mx.bool_)
#     for s in range(S - 1, -1, -1):
#         p = scale_E[s] / safe_total
#         cum = cum + p
#         reached = (cum >= 0.85) & remaining
#         rolloff = mx.where(reached, f[s], rolloff)
#         remaining = remaining & (~reached)

#     feats = mx.stack([centroid, variance, skewness, kurtosis, rolloff], axis=-1)
#     mx.eval(feats)
#     return _z(feats[sy, sx], stats)


# def _circular_features(
#     gw, sy: mx.array, sx: mx.array, stats=None
# ) -> tuple[mx.array, tuple]:
#     """Per-scale circular variance (S columns) + global isotropy (1 column).
#     Returns (feats, stats); pass stats back to freeze the anchors."""
#     S = gw.scale_size
#     thetas = mx.array(gw.orientations)
#     E = gw.energy_tensor  # (S, O, H, W)

#     iso_cols: list[mx.array] = []
#     amps_pooled = E.sum(axis=0)  # (O, H, W)
#     for s in range(S):
#         amps = E[s]  # (O, H, W)
#         total = amps.sum(axis=0, keepdims=True)
#         amps_n = amps / mx.maximum(total, 1e-12)
#         C = (amps_n * mx.cos(2.0 * thetas).reshape(-1, 1, 1)).sum(axis=0)
#         Ss = (amps_n * mx.sin(2.0 * thetas).reshape(-1, 1, 1)).sum(axis=0)
#         R = mx.sqrt(C * C + Ss * Ss)
#         cv = 1.0 - R
#         mx.eval(cv)
#         iso_cols.append(cv[sy, sx].reshape(-1, 1))

#     iso = mx.concatenate(iso_cols, axis=1)
#     total_pooled = amps_pooled.sum(axis=0, keepdims=True)
#     amps_pooled_n = amps_pooled / mx.maximum(total_pooled, 1e-12)
#     Cg = (amps_pooled_n * mx.cos(2.0 * thetas).reshape(-1, 1, 1)).sum(axis=0)
#     Sg = (amps_pooled_n * mx.sin(2.0 * thetas).reshape(-1, 1, 1)).sum(axis=0)
#     Rg = mx.sqrt(Cg * Cg + Sg * Sg)
#     cv_global = (1.0 - Rg)[sy, sx].reshape(-1, 1)
#     return _z(mx.concatenate([iso, cv_global], axis=1), stats)


# def _orientation_features(gw, sy: mx.array, sx: mx.array) -> mx.array:
#     """Orientation-sensitive encoding: raw doubled-angle resultant components.

#     Returns (N, 2S + 2): per-scale (C, S) pairs + global (Cg, Sg), where
#     C = R·cos(2θ̄), S = R·sin(2θ̄) and θ̄ is the dominant orientation.

#     Using the components instead of the angle itself keeps the encoding
#     continuous at the wrap-around (Gabor orientations are π-periodic) and
#     makes isotropic pixels map to (0, 0) — the angle is only expressed
#     where anisotropy actually exists.

#     These components are z-normalised here so they carry equal weight
#     with the other feature groups in the concatenated GMM feature vector.
#     The raw components are bounded in [-1, 1] and zero is meaningful
#     (isotropy), but without normalisation their variance can be much
#     lower than the z-normalised groups, effectively down-weighting
#     orientation information in a Euclidean-distance-based GMM.
#     """
#     S = gw.scale_size
#     thetas = mx.array(gw.orientations)
#     E = gw.energy_tensor  # (S, O, H, W)

#     cols: list[mx.array] = []
#     for s in range(S):
#         amps = E[s]  # (O, H, W)
#         total = amps.sum(axis=0, keepdims=True)
#         amps_n = amps / mx.maximum(total, 1e-12)
#         C = (amps_n * mx.cos(2.0 * thetas).reshape(-1, 1, 1)).sum(axis=0)
#         Ss = (amps_n * mx.sin(2.0 * thetas).reshape(-1, 1, 1)).sum(axis=0)
#         cols.append(C[sy, sx].reshape(-1, 1))
#         cols.append(Ss[sy, sx].reshape(-1, 1))

#     amps_pooled = E.sum(axis=0)  # (O, H, W)
#     total_pooled = amps_pooled.sum(axis=0, keepdims=True)
#     amps_pooled_n = amps_pooled / mx.maximum(total_pooled, 1e-12)
#     Cg = (amps_pooled_n * mx.cos(2.0 * thetas).reshape(-1, 1, 1)).sum(axis=0)
#     Sg = (amps_pooled_n * mx.sin(2.0 * thetas).reshape(-1, 1, 1)).sum(axis=0)
#     cols.append(Cg[sy, sx].reshape(-1, 1))
#     cols.append(Sg[sy, sx].reshape(-1, 1))

#     raw = mx.concatenate(cols, axis=1)
#     mx.eval(raw)
#     # z-normalise to balance weight with other feature groups in the GMM
#     return _z(raw)[0]


# def _dc_features(gw, sy: mx.array, sx: mx.array, stats=None) -> tuple[mx.array, tuple]:
#     """DC-channel features: local mean intensity + its 5×5 local std.
#     Returns (feats, stats); pass stats back to freeze the anchors.

#     Captures very-low-frequency structure (ramps, illumination gradients)
#     that the Gabor bank cannot reach — the coarsest wavelength is capped at
#     min(H, W)/2, so without this channel a slow ramp is indistinguishable
#     from flat. Side effect: clustering becomes illumination-sensitive.
#     """
#     dc = gw.dc_response
#     H, W = gw.H, gw.W
#     r = 2
#     p = mx.pad(dc, [(r, r), (r, r)], mode="edge")
#     n = (2 * r + 1) ** 2
#     a1 = mx.zeros((H, W), dtype=mx.float32)
#     a2 = mx.zeros((H, W), dtype=mx.float32)
#     for dy in range(2 * r + 1):
#         for dx in range(2 * r + 1):
#             w_ = p[dy : dy + H, dx : dx + W]
#             a1 = a1 + w_
#             a2 = a2 + w_ * w_
#     mean = a1 / n
#     std = mx.sqrt(mx.maximum(a2 / n - mean * mean, 0.0))
#     cols = mx.concatenate(
#         [dc[sy, sx].reshape(-1, 1), std[sy, sx].reshape(-1, 1)], axis=1
#     )
#     mx.eval(cols)
#     return _z(cols, stats)


# def _features(
#     gw,
#     sy: mx.array,
#     sx: mx.array,
#     orientation_sensitive: bool = False,
#     use_dc: bool = True,
#     use_pooled: bool = True,
#     use_circ: bool = True,
#     z_stats: dict | None = None,
# ) -> tuple[mx.array, dict]:
#     """Full luminance feature vector.

#     z_stats: optional dict of frozen per-group anchors (from a previous
#     call with return_stats=True) — keeps feature coordinates stable
#     across frames. Returns (feats, z_stats).
#     """
#     # use_pooled / use_circ are engine-level ablation switches (lesion
#     # experiments); the running system always leaves them on.
#     zs: dict = {} if z_stats is None else z_stats
#     out_stats: dict = {}
#     e, out_stats["energy"] = _scale_energy(gw, sy, sx, zs.get("energy"))
#     parts: list[mx.array] = [e]  # (N, S + 1)
#     if use_pooled:
#         p, out_stats["pooled"] = _pooled_spectral(gw, sy, sx, zs.get("pooled"))
#         parts.append(p)  # (N, 5)
#     if use_circ:
#         c, out_stats["circ"] = _circular_features(gw, sy, sx, zs.get("circ"))
#         parts.append(c)  # (N, S + 1)
#     s, out_stats["std"] = _local_std(gw, sy, sx, zs.get("std"))
#     parts.append(s)  # (N, 1)
#     if orientation_sensitive:
#         parts.append(_orientation_features(gw, sy, sx))  # (N, 2S + 2), z-normalised
#     if use_dc:
#         d, out_stats["dc"] = _dc_features(gw, sy, sx, zs.get("dc"))
#         parts.append(d)  # (N, 2)
#     return mx.concatenate(parts, axis=1), out_stats


# def _local_std(gw, sy: mx.array, sx: mx.array, stats=None) -> tuple[mx.array, tuple]:
#     """Illumination-relative local contrast: 3×3 local std normalised by
#     the local RMS (Weber-style gain control — a multiplicative
#     illumination field scales both equally). Point-local — minimal
#     straddling at boundaries. Flat pixels map to 0.
#     Returns (feats, stats) with stats = (z_mean, z_std, rms_mean) — pass
#     it back to freeze the anchors (temporal use).
#     """
#     img = gw.img
#     sq = img * img
#     pad = mx.pad(img, [(1, 1), (1, 1)], mode="edge")
#     pad_sq = mx.pad(sq, [(1, 1), (1, 1)], mode="edge")
#     mean = (
#         pad[2:, 1:-1]
#         + pad[:-2, 1:-1]
#         + pad[1:-1, 2:]
#         + pad[1:-1, :-2]
#         + pad[2:, 2:]
#         + pad[:-2, :-2]
#         + pad[2:, :-2]
#         + pad[:-2, 2:]
#         + img
#     ) / 9.0
#     mean_sq = (
#         pad_sq[2:, 1:-1]
#         + pad_sq[:-2, 1:-1]
#         + pad_sq[1:-1, 2:]
#         + pad_sq[1:-1, :-2]
#         + pad_sq[2:, 2:]
#         + pad_sq[:-2, :-2]
#         + pad_sq[2:, :-2]
#         + pad_sq[:-2, 2:]
#         + sq
#     ) / 9.0
#     var = mx.maximum(mean_sq - mean * mean, 0.0)
#     std = mx.sqrt(var)
#     rms = mx.sqrt(mean_sq)
#     if stats is None:
#         rmean = mx.maximum(rms.mean(), 1e-12)
#     else:
#         _zm, _zs, rmean = stats
#     eps = 0.01 * rmean
#     cv = std / (rms + eps)
#     mx.eval(cv)
#     z, zs = _z(cv[sy, sx].reshape(-1, 1), None if stats is None else stats[:2])
#     return z, (zs[0], zs[1], rmean)
