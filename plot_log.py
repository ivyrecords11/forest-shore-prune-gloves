# Program to load /mnt/data/LOG.csv, preview it, and automatically plot all numeric series
# against a reasonable x-axis (step/epoch/iteration/etc.).
#
# It saves:
#  - Per-metric PNGs in /mnt/data/plots/
#  - A single multi-page PDF report at /mnt/data/plots_report.pdf
#
# NOTE: No seaborn, single chart per figure, and no custom colors or styles are used.

import os
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages




def plot(csv_path: str):
    # 1) Load the CSV
    df = pd.read_csv(csv_path)

    # 3) Heuristics to pick the x-axis column
    #    Priorities: step/steps, epoch, episode, iter/iteration, time/t, frame, update
    priority_patterns = [
        r'^step$', r'^steps$',
        r'^epoch$', r'^epochs$',
        r'^episode$', r'^episodes$',
        r'^iter$', r'^iteration$', r'^iterations$',
        r'^time$', r'^t$', r'^timestamp$',
        r'^frame$', r'^frames$',
        r'^update$', r'^updates$'
    ]

    def pick_x_col(columns):
        lowered = [c.strip() for c in columns]
        # Exact matches by priority
        for pat in priority_patterns:
            for c in lowered:
                if re.match(pat, c, flags=re.IGNORECASE):
                    return c
        # If none, prefer the first integer-like column
        for c in lowered:
            s = pd.to_numeric(df[c], errors='coerce')
            if s.notna().all():
                # Check if it's integer-ish
                if np.all(np.floor(s.dropna()) == s.dropna()):
                    return c
        # Fallback: None (we'll use the index)
        return None

    x_col = pick_x_col(df.columns)

    # 4) Convert all columns to numeric where possible
    numeric_df = df.copy()
    for c in numeric_df.columns:
        numeric_df[c] = pd.to_numeric(numeric_df[c], errors='coerce')

    # 5) Determine y-columns
    if x_col is not None and x_col in numeric_df.columns:
        x = numeric_df[x_col]
        y_cols = [c for c in numeric_df.columns if c != x_col and pd.api.types.is_numeric_dtype(numeric_df[c])]
    else:
        x = pd.Series(np.arange(len(numeric_df)), name="index")
        y_cols = [c for c in numeric_df.columns if pd.api.types.is_numeric_dtype(numeric_df[c])]

    # Remove columns with all NaNs or identical values
    clean_y_cols = []
    for c in y_cols:
        series = numeric_df[c].dropna()
        if series.empty:
            continue
        # Skip if all values are the same
        if series.nunique() <= 1:
            continue
        clean_y_cols.append(c)

    # 6) Prepare output paths
    pdf_path = f"{csv_path}/plots_report.pdf"

    # 7) Compute a reasonable smoothing window (rolling mean) for noisy curves
    n = len(numeric_df)
    window = max(5, int(n * 0.05))  # 5% of length, at least 5
    # Cap the window so it doesn't get too large
    window = min(window, max(5, n // 5))  # at most 20% of series

    # 8) Create plots
    plots_made = []
    with PdfPages(pdf_path) as pdf:
        for c in clean_y_cols:
            y = numeric_df[c]

            # Create a single-figure chart (no subplots); do not set colors or styles.
            fig, ax = plt.subplots(figsize=(9, 5))

            if x_col is not None and x.notna().any():
                ax.plot(x, y, label=c)
                # Smoothed line (rolling mean) if window valid and series long enough
                if n >= window and window >= 5:
                    y_smooth = y.rolling(window=window, min_periods=max(2, window // 3)).mean()
                    ax.plot(x, y_smooth, label=f"{c} (rolling mean, w={window})")
                ax.set_xlabel(x_col)
            else:
                ax.plot(y.values, label=c)
                if n >= window and window >= 5:
                    y_smooth = y.rolling(window=window, min_periods=max(2, window // 3)).mean()
                    ax.plot(y_smooth.values, label=f"{c} (rolling mean, w={window})")
                ax.set_xlabel("index")

            ax.set_ylabel(c)
            ax.set_title(f"{c} over {x_col if x_col is not None else 'index'}")
            ax.legend()
            fig.tight_layout()

            # Save PNG and append to multi-page PDF
            png_path = f"{csv_path}/{c}.png"
            fig.savefig(png_path, dpi=150)
            pdf.savefig(fig)
            plots_made.append((c, png_path))
            plt.close(fig)

    # Report summary for the notebook output
    {"x_column_used": x_col, "num_plots": len(plots_made), "pdf_report": pdf_path, "example_png": plots_made[0][1] if plots_made else None}


plot("/TRAIN_DQN/1_train\LOG.csv")
