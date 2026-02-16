import pandas as pd
import os
import ast


def clean_accuracy_value(val):
    """
    Parses a string dictionary like "{'accuracy': 0.95}" and returns 0.95.
    Returns the original value if parsing fails or key is missing.
    """
    if pd.isna(val):
        return val

    try:
        if isinstance(val, str):
            data = ast.literal_eval(val)
        else:
            data = val

        if isinstance(data, dict) and "accuracy" in data:
            return data["accuracy"]
        return val
    except (ValueError, SyntaxError):
        return val


def extract_specific_columns(file_path):
    """
    Loads the experiment results and extracts QFrCoLA and FrCoE accuracy columns,
    cleaning dictionary strings into float values.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"Loading {file_path}...")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Define the columns we want to extract
    target_columns = ["name", "qfrcola.accuracy", "frcoe.accuracy"]

    # Verify columns exist
    available_columns = [col for col in target_columns if col in df.columns]
    missing_columns = [col for col in target_columns if col not in df.columns]

    if missing_columns:
        print(
            f"Warning: The following columns were not found in the file: {missing_columns}"
        )

    if not available_columns:
        print("No matching columns found to extract.")
        return

    # Extract the data
    filtered_df = df[available_columns].copy()

    # Clean the dictionary columns
    cols_to_clean = ["qfrcola.accuracy", "frcoe.accuracy"]

    print("Cleaning dictionary values...")
    for col in cols_to_clean:
        if col in filtered_df.columns:
            filtered_df[col] = filtered_df[col].apply(clean_accuracy_value)

    # --- SPECIFIC MANIPULATION: Transfer unsloth value to openai and delete unsloth ---
    source_name = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
    target_name = "openai/gpt-oss-20b"

    source_mask = filtered_df["name"] == source_name
    target_mask = filtered_df["name"] == target_name

    if source_mask.any() and target_mask.any():
        print(
            f"\nProcessing manual override: Transferring qfrcola from '{source_name}' to '{target_name}'"
        )

        source_value = filtered_df.loc[source_mask, "qfrcola.accuracy"].values[0]
        filtered_df.loc[target_mask, "qfrcola.accuracy"] = source_value

        print(f"Deleting row '{source_name}'")
        filtered_df = filtered_df[~source_mask]
    else:
        print(
            f"\nSkipping manual override: Source or Target row not found ({source_name} -> {target_name})"
        )

    # Remove CohereForAI/aya-expanse-8b
    row_to_remove = "CohereForAI/aya-expanse-8b"
    if (filtered_df["name"] == row_to_remove).any():
        print(f"Deleting row '{row_to_remove}'")
        filtered_df = filtered_df[filtered_df["name"] != row_to_remove]

    # Print the result to console
    print("\nExtracted Data Preview:")
    print(filtered_df.head())
    print(f"\nTotal rows extracted: {len(filtered_df)}")

    # --- Check for rows missing entries in either column ---
    check_cols = [c for c in cols_to_clean if c in filtered_df.columns]

    if check_cols:
        missing_mask = filtered_df[check_cols].isna().any(axis=1)
        missing_rows = filtered_df[missing_mask]

        if not missing_rows.empty:
            print(
                f"\n--- WARNING: Found {len(missing_rows)} rows with missing values ---"
            )

            for index, row in missing_rows.iterrows():
                missing_columns_in_row = [
                    col for col in check_cols if pd.isna(row[col])
                ]
                row_name = row["name"] if "name" in row else f"Index {index}"
                print(
                    f"Row '{row_name}' is missing: {', '.join(missing_columns_in_row)}"
                )
        else:
            print("\nAll extracted rows have complete data in both columns.")

    # Save to a new file
    output_filename = os.path.join("results", "qfrcola_filtered_accuracies.csv")
    filtered_df.to_csv(output_filename, index=False)
    print(f"\nSaved extracted data to '{output_filename}'")


if __name__ == "__main__":
    INPUT_FILE = os.path.join("results", "combined_experiment_results.csv")
    extract_specific_columns(INPUT_FILE)
