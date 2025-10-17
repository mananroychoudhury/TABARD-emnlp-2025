import os

def rename_json_files(directory: str):
    """
    Rename all .json files in `directory` by appending '_updated' to the basename.
    e.g. 'data.json' → 'data_updated.json'
    """
    # List everything in the directory
    for fname in os.listdir(directory):
        # Only target .json files
        if fname.lower().endswith('.json'):
            old_path = os.path.join(directory, fname)
            base, ext = os.path.splitext(fname)
            new_fname = f"{base}_updated{ext}"
            new_path = os.path.join(directory, new_fname)

            # Avoid overwriting an existing file
            if os.path.exists(new_path):
                print(f"Skipping {fname}: {new_fname} already exists.")
                continue

            os.rename(old_path, new_path)
            print(f"Renamed '{fname}' → '{new_fname}'")

if __name__ == "__main__":
    # Example usage
    target_folder = r"path_to_dataset\wikiTQ-org\Value_Anomaly_WikiTQ"
    rename_json_files(target_folder)
