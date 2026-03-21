"""
Colorblind-safe palette for MIUA 2026 figures.
Based on the Okabe-Ito / Wong palette, verified with Coblis simulator.

Usage:
    from colors import STRATEGY_COLORS, OVERLAY_COLORS, MARKERS
"""

# --- Adaptation strategy colors (Okabe-Ito palette) ---
STRATEGY_COLORS = {
    "frozen":   "#0072B2",  # blue
    "partial":  "#E69F00",  # orange
    "full":     "#D55E00",  # vermillion
    "bitfit":   "#56B4E9",  # sky blue
    "adapter":  "#CC79A7",  # reddish purple
    "baseline": "#000000",  # black
}

# --- Contrastive pair colors ---
CONTRAST = {
    "peft":     "#0072B2",  # blue (preservation)
    "full_ft":  "#D55E00",  # vermillion (destruction)
    "baseline": "#009E73",  # bluish green
}

# --- Segmentation overlay colors (RGB, 0-1 scale) ---
# Avoids red/green; uses blue/orange/purple instead
OVERLAY_COLORS = {
    "tp": (0.0, 0.45, 0.70),    # blue
    "fn": (0.90, 0.62, 0.0),    # orange
    "fp": (0.80, 0.47, 0.65),   # reddish purple
}
OVERLAY_COLORS_RGBA = {
    "tp": (0.0, 0.45, 0.70, 0.6),
    "fn": (0.90, 0.62, 0.0, 0.8),
    "fp": (0.80, 0.47, 0.65, 0.7),
}

# --- ASOCA k-value colors ---
K_COLORS = {
    1:  "#56B4E9",  # sky blue
    5:  "#009E73",  # bluish green
    10: "#E69F00",  # orange
}

# --- Markers for weight/model variants ---
MARKERS = {
    "baseline": "*",
    "w003":     "o",
    "w005":     "s",
    "w010":     "D",
    "w050":     "^",
}

# --- Bar chart colors ---
BAR_COLORS = {
    "dice": "#0072B2",  # blue
    "bcs":  "#E69F00",  # orange
}

# --- Annotation colors ---
ANNOT = {
    "highlight": "#F0E442",  # yellow (high contrast on dark bg)
    "circle":    "#F0E442",
}
