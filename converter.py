import pickle as pkl
import pandas as pd
with open("D:/BCRL_data/stringhe_mapper1", "rb") as f:
    object = pkl.load(f)
    
df = pd.DataFrame(object)
df.to_csv(r'stringhe_mapper1.csv',header=False, index=False)
