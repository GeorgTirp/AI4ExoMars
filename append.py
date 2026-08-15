import pandas as pd
import glob

df1 = pd.read_csv('/home/gtirpitz/AI4ExoMars/hirise_context_pairs/patch_index.csv')
df2 = pd.read_csv('/home/gtirpitz/AI4ExoMars/hirise_context_pairs_W2/patch_index.csv')
df3 = pd.read_csv('/home/gtirpitz/AI4ExoMars/hirise_context_pairs_W3/patch_index.csv')

#append the three dataframes together
df = pd.concat([df1, df2, df3], ignore_index=True)

#safe it in /data
df.to_csv('/home/gtirpitz/AI4ExoMars/data/patch_index.csv', index=False)
