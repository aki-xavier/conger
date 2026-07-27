"""Monocular feature-layer demo — end-to-end pipeline entry point.

Usage
-----
    uv run python main.py                          # run all tests
    uv run python main.py --signal 01              # test a single synthetic signal
    uv run python main.py --signal 01 --visualize  # test + save visualizations
    uv run python main.py --image path/to/img.png  # process a real image
    uv run python main.py --info                   # print layer configuration
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

# Add src/ to path for direct invocation
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gabor import GaborWavelet
from monocular import MonocularFeatureLayer, MonocularFeatures
from utils import Utils

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _print_header(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def _print_features(result: MonocularFeatures) -> None:
    """Print diagnostic information about extracted features."""
    feats = result.features
    print(f"  Feature matrix   : {feats.shape}  (N={result.H * result.W}, D={result.dim})")
    print(f"  Image            : {result.H}×{result.W}")
    print(f"  Scales           : {result.gw.scale_size}")
    print(f"  Orientations     : {result.gw.orientation_size}")
    print(f"  Wavelengths      : {[f'{w:.1f}' for w in result.gw.wavelengths]}")
    print("  Feature stats    :")
    for group, stats in result.z_stats.items():
        if isinstance(stats, tuple):
            mean_arr = stats[0]
            std_arr = stats[1]
        else:
            mean_arr = stats
            std_arr = mx.array(0.0)
        if hasattr(mean_arr, 'shape') and mean_arr.size > 0:
            print(f"    {group:8s}  mean={float(mean_arr.mean().item()):.3f}  "
                  f"std={float(std_arr.mean().item() if std_arr.size > 0 else 0):.3f}  "
                  f"dim={mean_arr.shape[-1] if mean_arr.ndim > 0 else 1}")
    print(f"  Orientation enc. : {result.coherence.get('orientation_enabled', False)}")
    print(f"  Chromatic enc.   : {result.coherence.get('chroma_enabled', False)}")
    print(f"  Dir / iso ratio  : {result.coherence.get('dir_iso_ratio', 0):.3f}")
    print(f"  Chroma coh gain  : {result.coherence.get('chroma_coh_gain', 0):.3f}")
    if result.bmap is not None:
        print(f"  Boundary channel : enabled  (max={float(result.bmap.max().item()):.3f})")
    if result.chromatic is not None:
        print(f"  Chromatic enc.   : enabled  shape=({result.H},{result.W})")


def _run_signal(name: str, signal_array: mx.array, visualize: bool = False) -> MonocularFeatures | None:
    """Run the monocular feature layer on one synthetic test signal."""
    if signal_array.ndim == 2:
        H, W_sig = signal_array.shape
        rgb = mx.stack([signal_array] * 3, axis=-1)
    else:
        H, W_sig = signal_array.shape[:2]
        rgb = signal_array
    # Ensure H >= 64 for sufficient scale resolution
    if H < 64 or W_sig < 64:
        print("  SKIP: signal too small (min 64 px)")
        return None
    mf = MonocularFeatureLayer(
        orientation_sensitive=None,  # auto-decide
        use_dc=True,
        boundary_radius=4,
    )
    t0 = time.perf_counter()
    result = mf.extract(rgb)
    t1 = time.perf_counter()
    mx.eval(result.features)
    if result.bmap is not None:
        mx.eval(result.bmap)
    t2 = time.perf_counter()

    _print_header(f"Signal {name}")
    print(f"  Extract time     : {t2 - t0:.3f} s  (MLX eval: {t2 - t1:.3f} s)")
    _print_features(result)

    if visualize:
        _visualize(result, rgb, f"signal{name}")
    return result


def _run_image(path: str, visualize: bool = False) -> None:
    """Run the monocular feature layer on a real image file."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    rgb = mx.array(arr, dtype=mx.float32)
    H, W = rgb.shape[:2]

    mf = MonocularFeatureLayer(
        orientation_sensitive=None,
        use_dc=True,
        boundary_radius=4,
    )
    t0 = time.perf_counter()
    result = mf.extract(rgb)
    t1 = time.perf_counter()
    mx.eval(result.features)
    if result.bmap is not None:
        mx.eval(result.bmap)
    t2 = time.perf_counter()

    _print_header(f"Real image: {path}")
    print(f"  Size             : {H}×{W}")
    print(f"  Extract time     : {t2 - t0:.3f} s  (MLX eval: {t2 - t1:.3f} s)")
    _print_features(result)

    if visualize:
        stem = Path(path).stem
        _visualize(result, rgb, stem)


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════


def _visualize(result: MonocularFeatures, rgb: mx.array, name: str) -> None:
    """Render and save a multi-panel visualization of the feature layer output.

    Panels
    ------
    Row 1: Input (luminance)  |  DC response  |  Boundary map  |  Bimodality
    Row 2: Gabor energy — selected scales × orientations
    Row 3: Feature heatmaps — spectral shape, spectral moments, circular
           variance, local contrast
    Row 4: Feature correlation matrix
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    out_dir = Utils.out_dir("visualizations")
    out_path = out_dir / f"{name}.png"

    H, W = result.H, result.W
    S = result.gw.scale_size
    O = result.gw.orientation_size
    luma = (
        np.array(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
        if rgb.ndim == 3
        else np.array(rgb)
    )

    # ── helpers ──────────────────────────────────────────────────
    def _np(a: mx.array) -> np.ndarray:
        return np.array(a)

    def _reshape_to_img(a: mx.array) -> np.ndarray:
        """Reshape (N,) or (N, 1) feature column to (H, W)."""
        arr = np.array(a).ravel()
        if arr.shape[0] == H * W:
            return arr.reshape(H, W)
        return arr[: H * W].reshape(H, W)
    # ── figure ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(5, 4, figure=fig, hspace=0.40, wspace=0.30)

    # Row 1: Input, DC, bmap, bimod ───────────────────────────────
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(luma, cmap="gray", aspect="auto")
    ax_img.set_title("Input (luminance)", fontsize=9)
    ax_img.axis("off")

    ax_dc = fig.add_subplot(gs[0, 1])
    dc_img = _np(result.gw.dc_response)
    ax_dc.imshow(dc_img, cmap="gray", aspect="auto")
    ax_dc.set_title("DC response", fontsize=9)
    ax_dc.axis("off")

    ax_bmap = fig.add_subplot(gs[0, 2])
    if result.bmap is not None:
        ax_bmap.imshow(_np(result.bmap), cmap="hot", aspect="auto", vmin=0, vmax=1)
        ax_bmap.set_title(f"Boundary (Pb)  max={float(result.bmap.max().item()):.2f}", fontsize=9)
    else:
        ax_bmap.text(0.5, 0.5, "disabled", ha="center", va="center", transform=ax_bmap.transAxes)
        ax_bmap.set_title("Boundary (Pb)", fontsize=9)
    ax_bmap.axis("off")

    ax_bimod = fig.add_subplot(gs[0, 3])
    if result.bimod is not None:
        ax_bimod.imshow(_np(result.bimod), cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax_bimod.set_title("Junction bimodality", fontsize=9)
    else:
        ax_bimod.text(0.5, 0.5, "disabled", ha="center", va="center", transform=ax_bimod.transAxes)
        ax_bimod.set_title("Junction bimodality", fontsize=9)
    ax_bimod.axis("off")

    # Rows 2-3: Gabor energy — 3 scales × 4 orientations ──────────
    scales_show = [0, S // 2, S - 1]
    oris_show = [0, O // 4, O // 2, 3 * O // 4]
    E = result.gw.energy_tensor  # (S, O, H, W)
    for si, s in enumerate(scales_show):
        for oi, o in enumerate(oris_show):
            ax = fig.add_subplot(gs[1 + si, oi])
            e_map = np.log1p(_np(E[s][o]))
            ax.imshow(e_map, cmap="inferno", aspect="auto")
            lam = result.gw.wavelengths[s]
            theta_deg = result.gw.orientations[o] * 180 / np.pi
            ax.set_title(f"λ={lam:.0f} px  θ={theta_deg:.0f}°", fontsize=7)
            ax.axis("off")

    # Rows 4-5: Feature heatmaps ──────────────────────────────────
    feats = result.features  # (N, D)
    S_dim = S + 1  # spectral shape + log mag
    mom_start = S_dim
    circ_start = mom_start + 5
    ctr_start = circ_start + S_dim

    # Spectral shape (fine scale), mid scale, log magnitude
    ax_s0 = fig.add_subplot(gs[3, 0])
    ax_s0.imshow(_reshape_to_img(feats[:, 0]), cmap="coolwarm", aspect="auto")
    ax_s0.set_title(f"Spect. shape  λ={result.gw.wavelengths[0]:.0f}", fontsize=8)
    ax_s0.axis("off")

    ax_sm = fig.add_subplot(gs[3, 1])
    ax_sm.imshow(_reshape_to_img(feats[:, S // 2]), cmap="coolwarm", aspect="auto")
    ax_sm.set_title(f"Spect. shape  λ={result.gw.wavelengths[S // 2]:.0f}", fontsize=8)
    ax_sm.axis("off")

    ax_lm = fig.add_subplot(gs[3, 2])
    ax_lm.imshow(_reshape_to_img(feats[:, S]), cmap="coolwarm", aspect="auto")
    ax_lm.set_title("Log magnitude", fontsize=8)
    ax_lm.axis("off")

    # Circular variance (global)
    ax_circ = fig.add_subplot(gs[3, 3])
    ax_circ.imshow(_reshape_to_img(feats[:, circ_start + S]), cmap="magma", aspect="auto")
    ax_circ.set_title("Global circ. variance", fontsize=8)
    ax_circ.axis("off")

    # Spectral centroid, rolloff, local contrast, correlation
    ax_cent = fig.add_subplot(gs[4, 0])
    ax_cent.imshow(_reshape_to_img(feats[:, mom_start]), cmap="plasma", aspect="auto")
    ax_cent.set_title("Spectral centroid", fontsize=8)
    ax_cent.axis("off")

    ax_roll = fig.add_subplot(gs[4, 1])
    ax_roll.imshow(_reshape_to_img(feats[:, mom_start + 4]), cmap="plasma", aspect="auto")
    ax_roll.set_title("Spectral rolloff", fontsize=8)
    ax_roll.axis("off")

    ax_ctr = fig.add_subplot(gs[4, 2])
    ax_ctr.imshow(_reshape_to_img(feats[:, ctr_start]), cmap="RdYlBu_r", aspect="auto")
    ax_ctr.set_title("Local contrast (Weber)", fontsize=8)
    ax_ctr.axis("off")

    # Feature correlation matrix
    ax_corr = fig.add_subplot(gs[4, 3])
    n_sample = min(2000, feats.shape[0])
    idx = np.random.RandomState(42).choice(feats.shape[0], n_sample, replace=False)
    feat_sample = np.array(feats)[idx]
    corr = np.corrcoef(feat_sample.T)
    im_corr = ax_corr.imshow(corr, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax_corr.set_title(f"Feature correlation (D={result.dim})", fontsize=8)
    plt.colorbar(im_corr, ax=ax_corr, fraction=0.046, pad=0.02)

    # ── save ─────────────────────────────────────────────────────
    fig.suptitle(
        f"Monocular Feature Layer — {name}  |  "
        f"{H}×{W}  S={S} O={O}  D={result.dim}  "
        f"ori={'on' if result.coherence.get('orientation_enabled') else 'off'}  "
        f"chroma={'on' if result.coherence.get('chroma_enabled') else 'off'}",
        fontsize=10,
        y=0.99,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Visualization    : {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monocular feature-layer demo — end-to-end pipeline entry point."
    )
    parser.add_argument(
        "--signal",
        type=str,
        default=None,
        help="Synthetic signal number (01-09) or 'all' for all signals.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a real RGB image file.",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print monocular feature-layer configuration and exit.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization figures to visualizations/ directory.",
    )
    args = parser.parse_args()

    vis = args.visualize

    if args.info:
        _print_info()
        return

    if args.image:
        _run_image(args.image, visualize=vis)
        return

    if args.signal == "all":
        _run_all_signals(visualize=vis)
        return

    if args.signal is not None:
        _run_signal_named(args.signal, visualize=vis)
        return

    # Default: run signal 01 as a quick sanity check, then signal 09
    _print_header("DEFAULT — Quick sanity check: signal 01 + signal 09")
    _run_signal_named("01", visualize=vis)
    _run_signal_named("09", visualize=vis)


def _run_signal_named(num: str, visualize: bool = False) -> None:
    """Run on a single named signal."""
    fn = getattr(Utils, f"synthesize_signal{num}", None)
    if fn is None:
        print(f"Unknown signal: {num}")
        return
    size = 256
    arr = fn(size)
    if isinstance(arr, tuple):
        arr = arr[0]
    _run_signal(num, arr, visualize=visualize)


def _run_all_signals(visualize: bool = False) -> None:
    """Run the monocular feature layer on all nine synthetic signals."""
    signals = [
        ("01", Utils.synthesize_signal01),
        ("02", Utils.synthesize_signal02),
        ("03", Utils.synthesize_signal03),
        ("04", Utils.synthesize_signal04),
        ("05", Utils.synthesize_signal05),
        ("06", Utils.synthesize_signal06),
        ("07", Utils.synthesize_signal07),
        ("08", Utils.synthesize_signal08),
        ("09", Utils.synthesize_signal09),
    ]
    size = 256
    for name, fn in signals:
        arr = fn(size)
        if isinstance(arr, tuple):
            arr = arr[0]
        _run_signal(name, arr, visualize=visualize)


def _print_info() -> None:
    """Print the monocular feature-layer configuration."""
    H, W = 128, 256
    S = GaborWavelet.default_scale_size(H, W)
    base_dim = 2 * S + 8
    ori_dim = 2 * S + 2
    dc_dim = 2

    print("MonocularFeatureLayer — configuration reference")
    print("=" * 50)
    print(f"  Default image size    : {H}×{W}")
    print(f"  Auto scale count      : {S}")
    print(f"  Base feature dim      : {base_dim}  (spectral shape {S}+1 + 5 moments + {S}+1 circ + 1 contrast)")
    print(f"  + orientation (2θ)    : {ori_dim}  (total: {base_dim + ori_dim})")
    print(f"  + DC channel          : {dc_dim}    (total: {base_dim + ori_dim + dc_dim})")
    print("  Gabor params          : λ_min=3.0 px, octave-spaced, 8 orientations")
    print("  Gain control          : divisive normalization + ε=1 % floor")
    print("  Boundary detector     : Pb half-disk, radius=4, 8 candidate θ")
    print("  Coherence criterion   : auto-selects orientation group (>15 % dir/iso)")
    print("                         auto-selects chromatic encoding (>15 % chroma gain)")


if __name__ == "__main__":
    main()
