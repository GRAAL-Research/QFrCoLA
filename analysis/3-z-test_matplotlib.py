import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os


def compute_comprehensive_z_test(file_path):
    print(f"Loading data from {file_path}...")

    if not os.path.exists(file_path):
        print(
            f"Error: File '{file_path}' not found. Please run the extraction script first."
        )
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
        print(f"Error: Columns '{col_left}' and '{col_right}' must exist in the CSV.")
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
        print(
            f"Baseline Scores - Académie française: {baseline_acad_fr:.2f}%, QFrCoLA: {baseline_qfrcola:.2f}%"
        )

        valid_data = all_data[all_data.index != baseline_row.index[0]].copy()
    else:
        print(
            "Warning: 'RandomBaselineModel' not found. Baseline comparison (Red) will be skipped/defaulted."
        )
        valid_data = all_data.copy()

    # --- CONFIGURATION ---
    n_left = 1651  # Académie française OOD test set size (ood/ood.tsv)
    n_right = 7546  # QFrCoLA test set size
    alpha = 0.001
    critical_z = stats.norm.ppf(1 - (alpha / 2))

    # ==========================================
    # PART 1: ROW-WISE CALCS & CLASSIFICATION
    # ==========================================
    print("Computing statistics and classifying...")

    results = []

    for index, row in valid_data.iterrows():
        p1_pct = row[col_left]
        p2_pct = row[col_right]
        model_name = row[name_col]

        # Z-test stats
        p1_prop = p1_pct / 100.0
        p2_prop = p2_pct / 100.0
        x1 = p1_prop * n_left
        x2 = p2_prop * n_right
        p_pool = (x1 + x2) / (n_left + n_right)

        if p_pool <= 0 or p_pool >= 1:
            se = 0
            z = 0
        else:
            se = np.sqrt(p_pool * (1 - p_pool) * ((1 / n_left) + (1 / n_right)))
            z = (p1_prop - p2_prop) / se if se != 0 else 0

        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        is_significant = p_val < alpha

        # --- CLASSIFICATION LOGIC ---
        is_below_baseline = False
        if has_baseline:
            if p1_pct < baseline_acad_fr or p2_pct < baseline_qfrcola:
                is_below_baseline = True

        if p1_pct > 65.0 and p2_pct > 65.0:
            classification = "High Performance (>65%)"
            color_code = "green"
        elif is_below_baseline:
            classification = "Below Random Baseline"
            color_code = "red"
        else:
            classification = "Intermediate"
            color_code = "blue"

        diff = p1_pct - p2_pct

        if not is_significant:
            direction_cat = "Not Significant"
        elif z > 0:
            direction_cat = "Acad. fr. > QFrCoLA"
        else:
            direction_cat = "QFrCoLA > Acad. fr."

        results.append(
            {
                "name": model_name,
                "acc_academie_francaise": p1_pct,
                "acc_qfrcola": p2_pct,
                "diff": diff,
                "z_score": z,
                "p_value": p_val,
                "significant": is_significant,
                "category": direction_cat,
                "Classification": classification,
                "Color": color_code,
            }
        )

    results_df = pd.DataFrame(results)
    csv_filename = os.path.join("figures_tables", "data", "qfrcola_row_wise_z_test_results.csv")
    results_df.to_csv(csv_filename, index=False)

    if results_df.empty:
        print("No results to plot.")
        return

    # ==========================================
    # PLOT 1: 3-COLOR SCATTER PLOT
    # ==========================================
    print("Generating Classified Scatter Plot...")

    sns.set_style("white")

    plt.figure(figsize=(12, 9))

    # Define Palette
    custom_palette = {
        "High Performance (>65%)": "green",
        "Below Random Baseline": "red",
        "Intermediate": "blue",
    }

    # Ensure Hue Order
    hue_order = ["High Performance (>65%)", "Intermediate", "Below Random Baseline"]
    present_categories = [
        cat for cat in hue_order if cat in results_df["Classification"].unique()
    ]

    sns.scatterplot(
        data=results_df,
        x="acc_academie_francaise",
        y="acc_qfrcola",
        hue="Classification",
        palette=custom_palette,
        hue_order=[c for c in hue_order if c in present_categories],
        alpha=0.5,
        s=80,
        edgecolor="white",
        linewidth=0.5,
        legend=False,
    )

    # 1. Diagonal Line (Equal Performance) -> Dash-Dot (-.)
    min_val = min(results_df["acc_academie_francaise"].min(), results_df["acc_qfrcola"].min())
    max_val = max(results_df["acc_academie_francaise"].max(), results_df["acc_qfrcola"].max())
    pad = (max_val - min_val) * 0.05
    limit_min = max(0, min_val - pad)
    limit_max = min(100, max_val + pad)

    plt.plot(
        [limit_min, limit_max],
        [limit_min, limit_max],
        color="black",
        linestyle="-.",
        alpha=0.4,
        linewidth=1,
        label="Equal Performance",
    )

    # 2. Baseline Lines -> Dashed (--)
    if has_baseline:
        plt.axvline(
            x=baseline_acad_fr,
            color="#444444",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="Random Baseline",
        )
        plt.axhline(
            y=baseline_qfrcola,
            color="#444444",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
        )

    # 3. Significance Bands (+/- Critical Z Interval)
    x_range = np.linspace(limit_min, limit_max, 300)
    x_prop = x_range / 100.0
    se_curve = np.sqrt(x_prop * (1 - x_prop) * ((1 / n_left) + (1 / n_right)))
    margin_curve = critical_z * se_curve * 100

    y_upper = x_range + margin_curve
    y_lower = x_range - margin_curve

    plt.plot(
        x_range,
        y_upper,
        color="gray",
        linestyle=":",
        alpha=0.5,
        linewidth=2.0,
        label=f"Sig. Interval (α={alpha})",
    )
    plt.plot(x_range, y_lower, color="gray", linestyle=":", alpha=0.5, linewidth=2.0)

    # Formatting
    plt.xlabel("Académie française Accuracy (%)")
    plt.ylabel("QFrCoLA Accuracy (%)")
    plt.xlim(limit_min, limit_max)
    plt.ylim(limit_min, limit_max)

    plt.grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join("figures_tables", "qfrcola_z_score_scatter_plot.png"), dpi=300)
    print("Saved qfrcola_z_score_scatter_plot.png")
    plt.close()


if __name__ == "__main__":
    INPUT_FILE = os.path.join("results", "qfrcola_filtered_accuracies.csv")
    compute_comprehensive_z_test(INPUT_FILE)
