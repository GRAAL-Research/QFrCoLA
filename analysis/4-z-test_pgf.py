import pandas as pd
import numpy as np
from scipy import stats
import os


def compute_and_generate_latex(file_path):
    print(f"Loading data from {file_path}...")

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    col_left = "academie_francaise.accuracy"
    col_right = "qfrcola.accuracy"
    name_col = "name"

    if col_left not in df.columns or col_right not in df.columns:
        print(f"Error: Columns '{col_left}' and '{col_right}' must exist.")
        return

    # Keep all data initially to find the baseline
    all_data = df.dropna(subset=[col_left, col_right]).copy()

    # --- TRANSFORM TO [0, 100] SCALE ---
    print("Converting accuracies to [0, 100] scale...")
    all_data[col_left] = all_data[col_left] * 100
    all_data[col_right] = all_data[col_right] * 100

    # --- EXTRACT BASELINE & CLEAN DATA ---
    baseline_row = all_data[
        all_data[name_col]
        .astype(str)
        .str.contains("RandomBaseline", case=False, na=False)
    ]

    baseline_acad_fr = 0
    baseline_qfrcola = 0
    has_baseline = False

    if not baseline_row.empty:
        baseline_acad_fr = baseline_row.iloc[0][col_left]
        baseline_qfrcola = baseline_row.iloc[0][col_right]
        has_baseline = True
        print(f"Found Baseline: {baseline_row.iloc[0][name_col]}")

        valid_data = all_data[all_data.index != baseline_row.index[0]].copy()
    else:
        print("Warning: 'RandomBaselineModel' not found.")
        valid_data = all_data.copy()

    # --- CONFIGURATION ---
    n_left = 2675  # Académie française OOD test set size
    n_right = 7546  # QFrCoLA test set size
    alpha = 0.001
    critical_z = stats.norm.ppf(1 - (alpha / 2))

    # ==========================================
    # CLASSIFICATION LOGIC
    # ==========================================
    print("Classifying data...")

    results = []

    for index, row in valid_data.iterrows():
        p1_pct = row[col_left]
        p2_pct = row[col_right]

        # Classification Logic
        is_below_baseline = False
        if has_baseline:
            if p1_pct < baseline_acad_fr or p2_pct < baseline_qfrcola:
                is_below_baseline = True

        if p1_pct > 65.0 and p2_pct > 65.0:
            tex_class = "green_dot"
        elif is_below_baseline:
            tex_class = "red_dot"
        else:
            tex_class = "blue_dot"

        results.append(
            {"acc_academie_francaise": p1_pct, "acc_qfrcola": p2_pct, "tex_class": tex_class}
        )

    results_df = pd.DataFrame(results)

    # ==========================================
    # PREPARE DATA FOR PGFPLOTS
    # ==========================================

    # 1. Save Scatter Data
    os.makedirs(os.path.join("figures_tables", "data"), exist_ok=True)

    scatter_csv = os.path.join("figures_tables", "data", "qfrcola_pgf_scatter_data.csv")
    results_df.to_csv(scatter_csv, index=False)
    print(f"Saved scatter data to {scatter_csv}")

    if results_df.empty:
        print("No valid data points to plot.")
        return

    # 2. Save Significance Curve Data
    min_val = min(results_df["acc_academie_francaise"].min(), results_df["acc_qfrcola"].min())
    max_val = max(results_df["acc_academie_francaise"].max(), results_df["acc_qfrcola"].max())
    pad = (max_val - min_val) * 0.05
    limit_min = max(0, min_val - pad)
    limit_max = min(100, max_val + pad)

    x_range = np.linspace(limit_min, limit_max, 200)
    x_prop = x_range / 100.0
    x_prop = np.clip(x_prop, 0, 1)

    se_curve = np.sqrt(x_prop * (1 - x_prop) * ((1 / n_left) + (1 / n_right)))
    margin_curve = critical_z * se_curve * 100

    y_upper = x_range + margin_curve
    y_lower = x_range - margin_curve

    sig_df = pd.DataFrame({"x": x_range, "y_upper": y_upper, "y_lower": y_lower})
    sig_csv = "qfrcola_pgf_sig_curves.csv"
    sig_df.to_csv(os.path.join("figures_tables", "data", sig_csv), index=False)
    print(f"Saved significance curves to {sig_csv}")

    # ==========================================
    # GENERATE LATEX FILE
    # ==========================================
    print("Generating LaTeX code...")

    tex_content = rf"""\documentclass[border=10pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}

\begin{{document}}
\begin{{tikzpicture}}
\begin{{axis}}[
    width=12cm, height=9cm,
    % Labels
    xlabel={{Acad\\'emie fran\\c{{c}}aise Accuracy (\%)}},
    ylabel={{QFrCoLA Accuracy (\%)}},
    % Limits
    xmin={limit_min:.2f}, xmax={limit_max:.2f},
    ymin={limit_min:.2f}, ymax={limit_max:.2f},
    % Styling
    grid=none, % No grid
    scatter/classes={{
        green_dot={{mark=*, draw=green!60!black, fill=green!60!black, opacity=0.5}},
        red_dot={{mark=*, draw=red!60!black, fill=red!60!black, opacity=0.5}},
        blue_dot={{mark=*, draw=blue!60!black, fill=blue!60!black, opacity=0.5}}
    }}
]

% 1. Equal Performance Line (Dash-Dot)
\addplot [black, dashdotted, domain={limit_min}:{limit_max}] {{x}};

% 2. Baseline Lines (Dashed, Dark Gray)
"""
    if has_baseline:
        tex_content += rf"""\draw [black!70, dashed, line width=1pt] (axis cs:{baseline_acad_fr}, {limit_min}) -- (axis cs:{baseline_acad_fr}, {limit_max});
\draw [black!70, dashed, line width=1pt] (axis cs:{limit_min}, {baseline_qfrcola}) -- (axis cs:{limit_max}, {baseline_qfrcola});
"""

    tex_content += rf"""
% 3. Significance Bands (Dotted, Bolder Gray)
\addplot [gray, dotted, line width=2pt] table [x=x, y=y_upper, col sep=comma] {{{sig_csv}}};
\addplot [gray, dotted, line width=2pt] table [x=x, y=y_lower, col sep=comma] {{{sig_csv}}};

% 4. Scatter Points
\addplot [scatter, only marks, scatter src=explicit symbolic]
    table [x=acc_academie_francaise, y=acc_qfrcola, meta=tex_class, col sep=comma] {{{scatter_csv}}};

\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""

    with open(os.path.join("figures_tables", "qfrcola_z_score_plot.tex"), "w") as f:
        f.write(tex_content)

    print(
        "Success! Generated 'z_score_plot.tex'. Compile this using pdflatex or lualatex."
    )


if __name__ == "__main__":
    INPUT_FILE = os.path.join("results", "qfrcola_filtered_accuracies.csv")
    compute_and_generate_latex(INPUT_FILE)
