import os

ROOT = r"results/final_study/final_thesis"  # Adjust path if needed

def scan_directory(root_dir):
    print(f"Scanning directory: {root_dir}\n")
    for dirpath, dirnames, filenames in os.walk(root_dir):
        level = dirpath.replace(root_dir, "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(dirpath)}/")
        subindent = " " * 4 * (level + 1)
        for f in sorted(filenames):
            if f.endswith(".csv"):
                print(f"{subindent}{f}")

if __name__ == "__main__":
    scan_directory(ROOT)
