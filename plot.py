import argparse
import csv
import os
#import altair as alt
#print(alt.__version__)
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

pd.set_option('display.max_rows', 2000)

def load_data(path, d):
    with open(path, 'r') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            model = row[0]
            resolution = row[1]
            iter_time = float(row[2])
            d['model'].append(model)
            d['resolution'].append(resolution)
            d['iter_time'].append(iter_time)
    return d

def fuse_data(baseline, new):
    fused = {'config': [], 'ratio': []}
    items = len(baseline['model'])
    assert items == len(new['model'])
    for idx in range(items):
        assert baseline['model'][idx] == new['model'][idx]
        assert baseline['resolution'][idx] == new['resolution'][idx]
        ratio = baseline['iter_time'][idx]/new['iter_time'][idx]
        fused['config'].append(baseline['model'][idx] + '_' + baseline['resolution'][idx])
        fused['ratio'].append(ratio)
    return fused

parser = argparse.ArgumentParser()
parser.add_argument('--new', required=True)
parser.add_argument('--baseline', required=True)
args = parser.parse_args()

assert os.path.exists(args.new)
assert os.path.exists(args.baseline)

baseline = {'model': [], 'resolution': [], 'iter_time': []}
new = {'model': [], 'resolution': [], 'iter_time': []}
baseline = load_data(args.baseline, baseline)
new = load_data(args.new, new)
fused = fuse_data(baseline, new)
fused_df = pd.DataFrame(fused)
fused_sorted_df = fused_df.sort_values(by='ratio', ascending=False)
print(fused_sorted_df)
#bar = alt.Chart(fused_sorted_df).mark_bar().encode(x='config:N', y='ratio:Q')
#rule = alt.Chart(fused_sorted_df).mark_rule(color='red').encode(y='1.0:Q')
output_name = f'{os.path.splitext(os.path.basename(args.baseline))[0]}vs{os.path.splitext(os.path.basename(args.new))[0]}.pdf'
#chart = (bar+rule).properties(width=600)
#chart.save(output_name)
ax = sns.barplot(x='config', y='ratio', data=fused_sorted_df)
plt.savefig(output_name)
