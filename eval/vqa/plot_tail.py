import seaborn as sns;
import os

sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_tail_for_one_model(alpha, accuracy, model_name='default'):
    data = {'Tail size': alpha, model_name: accuracy}
    df = pd.DataFrame(data, dtype=float)
    df = pd.melt(df, ['Tail size'], var_name="Models", value_name="Accuracy")
    ax = sns.lineplot(x="Tail size", y="Accuracy", hue="Models", style="Models", data=df, markers=False, ci=None)
    plt.xscale('log')
    plt.ylim(0, 100)
    save_dir = 'figures'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig('figures/tail_plot_%s.pdf' % model_name)
    plt.close()
