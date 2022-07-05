import pandas as pd
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--load_folder", type=str, required=True)
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=500000)
    args = parser.parse_args()

    os.makedirs(args.save_folder, exist_ok=True)
    chunk_id = 0
    for chunk in pd.read_csv(os.path.join(args.load_folder, args.dataset_name), chunksize=args.chunk_size):
        chunk.to_csv(os.path.join(args.save_folder, f'{args.dataset_name}_chunk{chunk_id}.csv'), index=False)
        chunk_id += 1

    # python Divide_large_dataset.py --dataset_name "feature_df_all_selected_with_ecg_20220210_rtfixed.csv" --load_folder "/mnt/group1/yilong/JET-Detection-Data" --save_folder "/mnt/group1/yilong/JET-Detection-Data/ecg-pat40-tch-sinus_jet"