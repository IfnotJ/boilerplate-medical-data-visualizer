import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = ((df['weight'] / (df['height'] / 100) ** 2) > 25).astype(int)

# 3
df['cholesterol'] = np.where(df['cholesterol'] > 1, 1, np.where(df['cholesterol'] == 1, 0, df['cholesterol']))
df['gluc'] = np.where(df['gluc'] <= 1, 0, np.where(df['gluc'] > 1, 1, df['gluc']))

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'],
    var_name='variable', value_name='value')


    # 6
    df_cat = sns.catplot(x='variable', hue='value', col='cardio', data=df_cat, kind='count', col_order=[0, 1], col_wrap=2, height=5, aspect=1.5, palette = 'dark:green');
    

    # 7

    

    # 8
    fig = df_cat.fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975)) & (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975))]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=int))



    # 14
    fig, ax = plt.subplots(figsize=(14, 6))

    # 15
    sns.heatmap(corr, annot=True, cmap='Spectral', mask=mask, fmt='.1f', vmin=-1, vmax=1);


    # 16
    fig.savefig('heatmap.png')
    return fig
