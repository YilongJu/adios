import pandas as pd
import argparse
import os
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--load_folder", type=str, required=True)
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=500000)
    parser.add_argument("--channel_ID", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.save_folder, exist_ok=True)
    chunk_id = 0
    st = time.time()
    for chunk in pd.read_csv(os.path.join(args.load_folder, args.dataset_name), chunksize=args.chunk_size):
        print(f"[Time {time.time() - st:.2f}] Saving chunk {chunk_id}...")
        if args.channel_ID > 0:
            chunk_tmp = chunk.query(f"channle_ID == {args.channel_ID}")
        else:
            chunk_tmp = chunk
        chunk_tmp.to_csv(os.path.join(args.save_folder, f'{args.dataset_name}_chunk{chunk_id}.csv'), index=False)
        print(f"[Time {time.time() - st:.2f}] Chunk {chunk_id} saved.")
        chunk_id += 1

    # python Divide_large_dataset.py --dataset_name "feature_df_all_selected_with_ecg_20220210_rtfixed.csv" --load_folder "/mnt/group1/yilong/JET-Detection-Data" --save_folder "/mnt/group1/yilong/JET-Detection-Data/ecg-pat40-tch-sinus_jet" --chunk_size 100000
    # python Divide_large_dataset.py --dataset_name "feature_df_all_selected_with_ecg_20220210_rtfixed.csv" --load_folder "/mnt/group1/yilong/JET-Detection-Data" --save_folder "/mnt/group1/yilong/JET-Detection-Data/ecg-pat40-tch-sinus_jet_lead2" --chunk_size 25000 --channel_ID 2
    # python Divide_large_dataset.py --dataset_name "feature_df_all_selected_with_ecg_20220210_rtfixed.csv" --load_folder "/mnt/scratch07/yilong" --save_folder "/mnt/scratch07/yilong/ecg-pat40-tch-sinus_jet" --chunk_size 100000
    # python Divide_large_dataset.py --dataset_name "feature_df_all_selected_with_ecg_20220210_rtfixed.csv" --load_folder "/mnt/scratch07/yilong" --save_folder "/mnt/scratch07/yilong/ecg-pat40-tch-sinus_jet_lead2" --chunk_size 25000 --channel_ID 2