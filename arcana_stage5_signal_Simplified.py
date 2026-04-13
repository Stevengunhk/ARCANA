import os
import pandas as pd

STAGE4A_DIR = r'C:\Arbion Research\Stage 4A stat arb engine'
OUTPUT_DIR  = r'C:\Arbion Research\Stage 5 signal blending'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read rolling signals — already regime-gated, elite pairs only
blended = pd.read_csv(
    os.path.join(STAGE4A_DIR, 'signals_gated_rolling.csv'),
    index_col=0, parse_dates=True)

blended.to_csv(os.path.join(OUTPUT_DIR, 'blended_signal.csv'))

print(f'✓ Stage 5 complete')
print(f'  Shape      : {blended.shape}')
print(f'  Date range : {blended.index[0].date()} to {blended.index[-1].date()}')
print(f'  Pairs      : {blended.shape[1]}')
print(f'  Pair names : {list(blended.columns)}')