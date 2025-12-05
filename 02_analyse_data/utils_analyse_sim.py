import random
from tqdm import tqdm
import os

def process_all_mat_files(input_dir, output_dir, process_func, do_phi_omega_a):
    """Generic function to process all .mat files in a folder."""
    mat_files = [f for f in os.listdir(input_dir) if f.endswith(".mat")]
    random.shuffle(mat_files)
    processed = 0
    skipped = 0

    for filename in tqdm(mat_files, desc=f"Processing .mat files from {input_dir}"):
        filepath = os.path.join(input_dir, filename)
        if process_func(filepath, output_dir, do_phi_omega_a):
            processed += 1
        else:
            skipped += 1

    print(f"\n✅ Done ({input_dir}): {processed} processed, {skipped} skipped.")