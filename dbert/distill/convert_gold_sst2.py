import pandas as pd
import sys


df = pd.read_csv(sys.argv[1])
df['index'] = list(range(len(df)))
df[['index', 'prediction']].to_csv(sys.argv[2], sep='\t', index=False)
