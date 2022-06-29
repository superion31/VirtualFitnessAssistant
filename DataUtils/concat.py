import glob
import pandas as pd

def concat_dfs(file):
    
    filenames = glob.glob(str(file)+"/*.csv")

    dfs = []
    for filename in filenames:
        dfs.append(pd.read_csv(filename))

    # Concatenate all data into one DataFrame
    big_df = pd.concat(dfs, ignore_index=True)
    big_df
    
    big_df.to_csv(r'data.csv', index=False) 
