import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=500000)
    args = parser.parse_args()

    chunk_id = 0
    for chunk in pd.read_csv(f"{args.dataset_name}", chunksize=args.chunk_size):
        chunk.to_csv(f'{args.dataset_name}_chunk{chunk_id}.csv', index=False)
        chunk_id += 1