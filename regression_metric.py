import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import pdb
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from pathlib import Path


def plot_regression_metric(csv_path, save_dir, flag):
    df = pd.read_csv(csv_path)
    model_name = get_model_name_from_file(csv_path)
    save_name = f"{os.path.splitext(os.path.basename(csv_path))[0]}.png"
    save_path = os.path.join(save_dir, save_name)

    df['groundtruth'] = df['true_dice']
    df['predict'] = df['preds_dices']
    df_filtered = df[df['groundtruth'] >= 0]  # GT>=0.75

    pearson_r = round(pearsonr(df_filtered['groundtruth'], df_filtered['predict'])[0], 3)
    spearman_r = round(spearmanr(df_filtered['groundtruth'], df_filtered['predict'])[0], 3)

    plot_and_save(df_filtered, model_name, pearson_r, spearman_r, save_path)


def get_model_name_from_file(file_name):
    file_name_lower = os.path.basename(file_name).lower()
    if 'medsam' in file_name_lower:
        return "MedSAM"
    elif 'sammed2d' in file_name_lower:
        return "SAM-Med2D"
    elif 'sam' in file_name_lower:
        return "SAM"
    elif 'hsnet' in file_name_lower:
        return "HSNet"
    else:
        return 'None'


def plot_and_save(df_filtered, model_name, pearson_r, spearman_r, save_path):
    sns.set(style="whitegrid", context="talk")
    plt.figure(figsize=(10.5, 8))
    sns.scatterplot(x='groundtruth', y='predict', data=df_filtered, color='royalblue', alpha=0.6)

    X = df_filtered['groundtruth'].values.reshape(-1, 1)
    y = df_filtered['predict'].values
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    plt.plot(df_filtered['groundtruth'], predictions, color='red', linestyle='--', linewidth=2)

    plt.xlabel('Groundtruth DSC', fontsize=25)
    plt.ylabel('Predicted DSC', fontsize=25)
    plt.title(f'{model_name}, Pearson: {pearson_r:.3f}, Spearman: {spearman_r:.3f}', fontsize=25)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_name', type=str, default='test_bbox_medsam_FLARE21_vit_base_224.csv', help='path to csv file')
    parser.add_argument('--result_path', type=str, default='./result/exp_2/test/new_epoch_30_vit_base_224_1/run_4/save')
    parser.add_argument('--flag', type=int, default=2, help="logits num")
    args = parser.parse_args()
    save_path = args.result_path
    args.result_path = os.path.join(args.result_path, args.csv_name)
    plot_regression_metric(args.result_path, save_path, args.flag)
