import pandas as pd
import os

data_dir = 'camchex/data'
splits = ['train.csv', 'development.csv', 'test.csv']

for split in splits:
    csv_path = os.path.join(data_dir, split)
    if os.path.exists(csv_path):
        print(f"Filtering {split}...")
        df = pd.read_csv(csv_path)
        
        # Check if the file actually exists inside camchex/images/...
        existing_rows = df['path'].apply(lambda p: os.path.exists(os.path.join('camchex', p)))
        
        filtered_df = df[existing_rows]
        print(f"  -> Kept {len(filtered_df)} out of {len(df)} images.")
        
        # Save the CSV to a new file instead of overwriting
        filtered_csv_path = csv_path.replace('.csv', '_filtered.csv')
        filtered_df.to_csv(filtered_csv_path, index=False)
        print(f"  -> Saved to {filtered_csv_path}")
