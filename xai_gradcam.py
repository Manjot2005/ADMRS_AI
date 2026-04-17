"""
xai_gradcam.py — Explainable AI: Grad-CAM for U-Net
Generates heatmaps showing which pixels most influenced the model's
deforestation prediction. Uses the captum library.

Usage:
    from xai_gradcam import generate_gradcam
    heatmap = generate_gradcam(model, tile_array)
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path


def generate_gradcam(model: nn.Module, tile: np.ndarray, device: torch.device = None) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for a single tile.

    Args:
        model   : Trained U-Net (segmentation_models_pytorch)
        tile    : (H, W, 4) float32 array [R,G,B,NIR] normalized 0-1
        device  : torch device

    Returns:
        heatmap : (H, W) float32 array [0,1], 1=most influential pixels
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    # Prepare input tensor (1, 4, H, W)
    x = torch.tensor(tile).permute(2, 0, 1).unsqueeze(0).float().to(device)
    x.requires_grad_(True)

    # Forward pass — get deforestation class (class 1) logits
    logits = model(x)                          # (1, 2, H, W)
    # Focus on deforested class score (sum over spatial)
    target = logits[:, 1, :, :].sum()

    # Backward pass
    model.zero_grad()
    target.backward()

    # Grad-CAM: gradients × activations at input
    gradients = x.grad.data.abs()              # (1, 4, H, W)
    # Average over channels, take spatial map
    heatmap = gradients.squeeze().mean(dim=0)  # (H, W)
    heatmap = heatmap.cpu().numpy()

    # Normalize to [0, 1]
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    return heatmap.astype(np.float32)


def overlay_heatmap_on_rgb(
    tile: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.55,
    colormap: str = "RdYlGn_r"
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on RGB image.

    Args:
        tile     : (H, W, 4) float32 [R,G,B,NIR]
        heatmap  : (H, W) float32 [0,1]
        alpha    : blend strength of heatmap overlay
        colormap : matplotlib colormap name

    Returns:
        blended  : (H, W, 3) uint8 RGB image
    """
    rgb = (tile[:, :, :3] * 255).clip(0, 255).astype(np.uint8)

    cmap = cm.get_cmap(colormap)
    heat_rgb = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)

    blended = (rgb * (1 - alpha) + heat_rgb * alpha).clip(0, 255).astype(np.uint8)
    return blended


def plot_xai_figure(
    tile: np.ndarray,
    mask: np.ndarray,
    heatmap: np.ndarray,
    save_path: Path = None
) -> plt.Figure:
    """
    Create a 3-panel XAI figure:
      Left  : RGB input image
      Center: AI prediction mask
      Right : Grad-CAM heatmap overlay

    Returns matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="#0a0f0a")
    fig.suptitle(
        "Explainable AI — Grad-CAM Analysis",
        color="#4ade80", fontsize=13, fontweight="bold", y=1.02
    )

    titles = ["Input (RGB + NIR)", "AI Prediction Mask", "Grad-CAM (Why flagged?)"]
    bg = "#0e1610"

    # Panel 1: RGB
    axes[0].imshow(tile[:, :, :3].clip(0, 1))
    axes[0].set_title(titles[0], color="#d1fae5", fontsize=10, pad=8)

    # Panel 2: Prediction mask
    import matplotlib.colors as mcolors
    cmap2 = mcolors.ListedColormap(["#166534", "#ef4444"])
    axes[1].imshow(mask, cmap=cmap2, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(titles[1], color="#d1fae5", fontsize=10, pad=8)

    # Panel 3: Grad-CAM overlay
    overlay = overlay_heatmap_on_rgb(tile, heatmap)
    axes[2].imshow(overlay)
    axes[2].set_title(titles[2], color="#d1fae5", fontsize=10, pad=8)

    # Add colorbar for heatmap
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Influence", color="#d1fae5", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#6b9e75")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#6b9e75", fontsize=7)

    for ax in axes:
        ax.set_facecolor(bg)
        ax.axis("off")

    fig.patch.set_facecolor("#0a0f0a")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight",
                    facecolor="#0a0f0a", edgecolor="none")

    return fig


def generate_demo_gradcam(tile_size: int = 256) -> tuple:
    """
    Generate synthetic Grad-CAM data for demo without a real model.
    Returns (tile, mask, heatmap) as numpy arrays.
    """
    rng = np.random.default_rng(42)

    # Synthetic tile
    tile = rng.random((tile_size, tile_size, 4)).astype(np.float32)
    # Add a deforested patch (high red, low NIR)
    tile[80:160, 80:160, 0] = 0.8   # Red high
    tile[80:160, 80:160, 3] = 0.15  # NIR low
    tile[80:160, 80:160, 1] = 0.4   # Green mid
    tile[80:160, 80:160, 2] = 0.3   # Blue mid

    # Synthetic mask
    mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
    mask[80:160, 80:160] = 1

    # Synthetic heatmap — high values where deforested
    heatmap = np.zeros((tile_size, tile_size), dtype=np.float32)
    # Add gaussian blob at deforested region
    y, x = np.mgrid[0:tile_size, 0:tile_size]
    cy, cx = 120, 120
    sigma = 40
    heatmap = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    # Add some noise elsewhere
    heatmap += rng.random((tile_size, tile_size)) * 0.15
    heatmap = (heatmap / heatmap.max()).astype(np.float32)

    return tile, mask, heatmap
