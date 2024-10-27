import os
import re
import argparse
import seaborn as sns
from loguru import logger
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import ptitprince as pt
from scipy.stats import wilcoxon, normaltest, kruskal
from statsmodels.stats.multitest import multipletests

LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12
PALETTE = ['red', 'darkblue']

# order_datasets_tum = [
#     # 'deepseg_lesion',
#     # 'stitched_straight\nmulti-channel', 'chunks_straight\nmulti-channel', 'chunks_native\nmulti-channel', 'stitched_native\nmulti-channel',
#     'stitched_straight\nregion-based', 'chunks_straight\nregion-based', 'chunks_native\nregion-based', 'stitched_native\nregion-based',
# ]
order_datasets_tum = {
     "Dataset901_tumMSChunksRegion": 'Chunks\nNative', 
     "Dataset902_tumMSStitchedRegion": 'Stitched\nNative', 
     "Dataset903_tumMSChunksStraightRegion": 'Chunks\nStraightened', 
     "Dataset904_tumMSStitchedStraightRegion": 'Stitched\nStraightened',
    #  "Dataset910_tumMSChunksPolyNYUAxialRegion": 'Chunks\nPolyNYU',
}

# order_datasets_deepseg_lesion = [
#     'stitched_straight\nregion-based', 'chunks_straight\nregion-based', 'chunks_native\nregion-based', 'stitched_native\nregion-based',
# ]
order_datasets_testing_large = [
    'chunks_native\ntum', 'chunks_native\ntum_polyNYU', # 'deepseg_lesion',
]

order_datasets_muc_vs_neuropoly = [
    'stitched_straight\ntum', 'chunks_straight\ntum', 'chunks_native\ntum', 'stitched_native\ntum',
    'chunks_native\ntum_neuropoly'
]


def get_parser():

    parser = argparse.ArgumentParser(description='Plot metrics per site')
    parser.add_argument('-i', type=str, required=True, nargs='+',
                        help='Path to the folders containing all folds from nnUNet training for each model to compare')
    parser.add_argument('-o', type=str, required=True,
                        help='Path to the output folder where the plots will be saved')
    # parser.add_argument('--v2', type=str, required=True,
    #                     help='Path to the folder containing metrics for each site')
    parser.add_argument('-test-site', type=str, required=True,
                        help='Results to compare from which site: `muc` (TUM MS) or `deepseg`')
    parser.add_argument('-pred-type', type=str, required=True,
                        help='Type of prediction to create plots for: `sc` (spinal cord segmentation) or `lesion`')
    parser.add_argument('-compare-across', type=str, required=True, choices=['tum', 'tum-poly'],
                        help='Compare the models performance across TUM datasets (tum) i.e. single site'
                         'or across TUM, Poly datasets i.e. two sites')

    return parser


def find_dataset_in_path(path):
    """Extracts model identifier from the given path.

    Args:
    path (str): Input path containing a model identifier.

    Returns:
    str: Extracted model identifier or None if not found.
    """
    pattern = r'Dataset\d{3}_\w+'
    match = re.search(pattern, path)
    if match:
        return match.group(0)
    elif 'DeepSeg' in path:
        return 'deepseg_lesion'
    else:
        return 'Unknown'
    # Find 'nnUNetTrainer' followed by the model name
    
    # elif 'Dataset909_tumMSChunksNeuropolyAxialRegion' in path:
    #         dataset = 'chunks_native\ntum_neuropoly'
    if 'Dataset910_tumMSChunksPolyNYUAxialRegion' in path:
            dataset = 'chunks_native\ntum_polyNYU'
        
    return dataset


def find_model_in_path(path):
    match = re.search(r'2d|3d_fullres', path)
    if match:
        return '2D' if '2d' in match.group(0) else '3D'


def find_filename_in_path(path):
    fname = path.split('/')[-1]
    return fname.split('.')[0]


def create_rainplot(args, df, metrics, path_figures, pred_type):
    """
    Create Raincloud plots (violionplot + boxplot + individual points)
    :param df: dataframe with segmentation metrics
    :param list_of_metrics: list of metrics to be plotted
    :param path_figures: path to the folder where the figures will be saved
    :param pred_type: type of prediction to create plots for; sc: spinal cord segmentation; lesion: lesion segmentation
    :param num_of_seeds: number of seeds (obtained from the number of input folders)
    :return:
    """

    # mpl.rcParams['font.family'] = 'Helvetica'
    if args.compare_across == 'tum-poly':
        order_datasets = order_datasets_testing_large
    else:
        order_datasets = order_datasets_tum

    for metric in metrics.keys():
        fig_size = (9, 5.5) # if pred_type == 'sc' else (9, 5.5)
        fig, ax = plt.subplots(figsize=fig_size)
        ax = pt.RainCloud(data=df,
                          x='dataset',
                          y=metric,
                          hue='model',
                          palette=PALETTE,
                          order=order_datasets.keys(), #if pred_type == 'sc' else METHODS_TO_LABEL_LESION.keys(),
                          dodge=True,       # move boxplots next to each other
                          linewidth=0,      # violionplot border line (0 - no line)
                          width_viol=.5,    # violionplot width
                          width_box=.3,     # boxplot width
                          rain_alpha=.7,    # individual points transparency
                          rain_s=2,         # individual points size
                          alpha=.7,         # violin plot transparency
                          box_showmeans=True,  # show mean value inside the boxplots
                          box_meanprops={'marker': '^', 'markerfacecolor': 'black', 'markeredgecolor': 'black',
                                         'markersize': '6'},
                          hue_order=['3D', '2D'],
                          )

        # TODO: include mean +- std for each boxplot above the mean value

        # Change boxplot opacity (.0 means transparent)
        # https://github.com/mwaskom/seaborn/issues/979#issuecomment-1144615001
        for patch in ax.patches:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .0))

        # Include number of subjects for each site into the legend
        handles, labels = ax.get_legend_handles_labels()
        for i, label in enumerate(labels):
            n = len(df[(df['dataset'] == list(order_datasets.keys())[0]) & (df['model'] == label)])
            labels[i] = f'{label} Model' + ' ($\it{n}$' + f' = {n})'
        # Since the figure contains violionplot + boxplot + scatterplot we are keeping only last two legend entries
        handles, labels = handles[-2:], labels[-2:]
        ax.legend(handles, labels, fontsize=TICK_FONT_SIZE, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)

        # Make legend box's frame color black and remove transparency
        legend = ax.get_legend()
        legend.legendPatch.set_facecolor('white')
        legend.legendPatch.set_edgecolor('black')

        # Remove spines
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(True)

        # Remove x-axis label
        ax.set_xlabel('')
        # Modify x-ticks labels
        ax.set_xticklabels(order_datasets.values(), #if pred_type == 'sc' else METHODS_TO_LABEL_LESION.values(),
                           fontsize=TICK_FONT_SIZE)
        # Increase y-axis label font size
        ax.set_ylabel(metric, fontsize=TICK_FONT_SIZE)
        # Increase y-ticks font size
        ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)

        # # Adjust y-lim for 'RelativeVolumeError' metric
        # if metric == 'RelativeVolumeError' and pred_type == 'sc':
        #     ax.set_ylim(-95, 62)
        # elif metric == 'RelativeVolumeError' and pred_type == 'lesion':
        #     ax.set_ylim(-125, 125)

        # Set title
        if pred_type == 'sc':
            ax.set_title(f'Test {metrics[metric]} for Spinal Cord Segmentation', fontsize=LABEL_FONT_SIZE)
        else:
            ax.set_title(f'Test {metrics[metric]} for Lesion Segmentation', fontsize=LABEL_FONT_SIZE)

        # Move grid to background (i.e. behind other elements)
        ax.set_axisbelow(True)
        # Add horizontal grid lines and change its opacity
        ax.yaxis.grid(True, alpha=0.3)
        # modify the y-axis ticks
        if pred_type == 'lesion' and metric != 'RelativeVolumeError':
            ax.set_yticks(np.arange(0, 1.1, 0.1))
        elif pred_type == 'sc' and metric != 'RelativeVolumeError':
            ax.set_yticks(np.arange(0.6, 1.025, 0.05))
        
        plt.tight_layout()

        # save figure
        fname_fig = os.path.join(path_figures, f'rainplot_{pred_type}_{metric}.png')
        plt.savefig(fname_fig, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f'Created: {fname_fig}')


def create_regplot_lesion_count(args,df, path_figures):
    """
    Create a seaborn lmplot for reference vs predicted lesion count comparing 2D and 3D models
    :param df: dataframe with segmentation metrics
    :param path_figures: path to the folder where the figures will be saved
    :return:
    """
    order_datasets = order_datasets_tum if args.compare_across == 'tum' else order_datasets_testing_large

    # Set up a 2x2 plotting grid (assuming 4 datasets)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Loop over each dataset and plot 2D and 3D models with regression lines
    for i, dataset in enumerate(df['dataset'].unique()):
        dataset_data = df[df['dataset'] == dataset]
        
        # Plot 2D model with regression line and 95% confidence interval
        sns.regplot(
            x='RefLesionsCount', y='PredLesionsCount', 
            data=dataset_data[dataset_data['model'] == '2D'], label='2D Model',
            ax=axes[i], scatter=True, scatter_kws={'color': PALETTE[1]},
            ci=95, line_kws={'color': PALETTE[1]}, marker='^'
        )
        
        # Plot 3D model with regression line and 95% confidence interval
        sns.regplot(
            x='RefLesionsCount', y='PredLesionsCount', 
            data=dataset_data[dataset_data['model'] == '3D'], label='3D Model',
            ax=axes[i], scatter=True, scatter_kws={'color': PALETTE[0]},
            ci=95, line_kws={'color': PALETTE[0]}, marker='o'
        )

        # Set x and y limits
        axes[i].set_xlim(-1, 16)
        axes[i].set_ylim(-1, 16)
        
        # increase font size for x and y ticks
        axes[i].tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
        # increase font size for x and y labels
        axes[i].set_xlabel('Reference Lesions Count', fontsize=LABEL_FONT_SIZE)
        axes[i].set_ylabel('Predicted Lesions Count', fontsize=LABEL_FONT_SIZE)
        
        # Add a dashed line to show perfect prediction line
        axes[i].plot([-1, 16], [-1, 16], 'k--', label='Ideal')
        
        # Set title, labels, and legend for each subplot
        title = order_datasets[dataset].replace('\n', ' ')
        axes[i].set_title(f'Dataset: {title}', fontsize=LABEL_FONT_SIZE+2, fontweight='bold')
        axes[i].set_xlabel('Reference Lesions Count')
        axes[i].set_ylabel('Predicted Lesions Count')
        axes[i].legend(fontsize=LABEL_FONT_SIZE)

    # add a title to the figure (bold)
    fig.suptitle('Reference vs Predicted Lesion Count across Datasets', fontsize=LABEL_FONT_SIZE+3, fontweight='bold')

    # Adjust layout to avoid overlap
    plt.tight_layout()
    # save figure
    fname_fig = os.path.join(path_figures, f'regplot_LesionCount.png')
    plt.savefig(fname_fig, dpi=300, bbox_inches='tight')
    plt.close()


def format_pvalue(p_value, decimal_places=3, include_space=False, include_equal=True):
    """
    Format p-value.
    If the p-value is lower than alpha, format it to "<0.001", otherwise, round it to three decimals

    :param p_value: input p-value as a float
    :param decimal_places: number of decimal places the p-value will be rounded
    :param include_space: include space or not (e.g., ' = 0.06')
    :param include_equal: include equal sign ('=') to the p-value (e.g., '=0.06') or not (e.g., '0.06')
    :return: p_value: the formatted p-value (e.g., '<0.05') as a str
    """
    if include_space:
        space = ' '
    else:
        space = ''

    # If the p-value is lower than alpha, return '<alpha' (e.g., <0.001)
    for alpha in [0.001, 0.01, 0.05]:
        if p_value < alpha:
            p_value = space + "<" + space + str(alpha)
            break
    # If the p-value is greater than 0.05, round it number of decimals specified by decimal_places
    else:
        if include_equal:
            p_value = space + '=' + space + str(round(p_value, decimal_places))
        else:
            p_value = space + str(round(p_value, decimal_places))

    return p_value


def compute_wilcoxon_test(df_concat, list_of_metrics):
    """
    Compute Wilcoxon signed-rank test (two related paired samples -- a same subject for nnunet_3d vs nnunet_2d)
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
    :param df_concat: dataframe containing all the data
    :param list_of_metrics: list of metrics to compute the Wilcoxon test for
    :return:
    """

    logger.info('')

    # Remove 'NbTestedLesions' and 'VolTestedLesions' from the list of metrics
    # list_of_metrics = [metric for metric in list_of_metrics if metric not in ['NbTestedLesions', 'VolTestedLesions']]

    # keep only the filenames for reference and prediction
    df_concat['reference'] = df_concat['reference'].apply(find_filename_in_path)
    df_concat['prediction'] = df_concat['prediction'].apply(find_filename_in_path)

    # Loop across sites
    for dataset in df_concat['dataset'].unique():
        # Loop across metrics
        for metric in list_of_metrics:
            # Reorder the dataframe
            df_2d = df_concat[(df_concat['model'] == '2D') & (df_concat['dataset'] == dataset)]
            df_3d = df_concat[(df_concat['model'] == '3D') & (df_concat['dataset'] == dataset)]

            # ensure that the two dataframes have the same number of rows
            assert len(df_2d) == len(df_3d), \
                f"Number of subjects for 2D and 3D models are different for {dataset}, paired tests cannot be performed"

            # Combine the two dataframes based on participant_id and seed. Keep only metric column
            df = pd.merge(df_2d[['prediction', 'fold', metric]],
                          df_3d[['prediction', 'fold', metric]],
                          on=['prediction', 'fold'],
                          suffixes=('_2d', '_3d'))

            # Print number of subjects
            logger.info(f'{metric}, {dataset}: Number of subjects: {len(df)}')

            # Run normality test
            stat, p = normaltest(df[metric + '_2d'])
            logger.info(f'{metric}, {dataset}: Normality test for nnUNet2D: formatted p{format_pvalue(p)}, '
                        f'unformatted p={p:0.6f}')
            stat, p = normaltest(df[metric + '_3d'])
            logger.info(f'{metric}, {dataset}: Normality test for nnUNet3D: formatted p{format_pvalue(p)}, '
                        f'unformatted p={p:0.6f}')

            # Compute Wilcoxon signed-rank test
            stat, p = wilcoxon(df[metric + '_2d'], df[metric + '_3d'], alternative='greater')
            logger.info(f'{metric}, {dataset}: Wilcoxon signed-rank test between nnUNet2D and nnUNet3D: '
                        f'formatted p{format_pvalue(p)}, unformatted p={p:0.6f}')

        logger.info('')


def compute_kruskal_wallis_test_across_tum_datasets(df_concat, list_of_metrics):
    """
    Compute Kruskal-Wallis H-test (non-parametric version of ANOVA)
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
    :param df_concat:
    :param list_of_metrics:
    :return:
    """
    logger.info('')

    # keep only the filenames for reference and prediction
    df_concat['reference'] = df_concat['reference'].apply(find_filename_in_path)
    df_concat['prediction'] = df_concat['prediction'].apply(find_filename_in_path)

    # Loop across sites
    # for site in df_concat['site'].unique():
    # Loop across metrics
    for metric in list_of_metrics:
        
        # Reorder the dataframe
        # 1. bavaria only models
        df_chunks_native_2d = df_concat[(df_concat['dataset'] == 'Dataset901_tumMSChunksRegion') & (df_concat['model'] == '2D')]
        # df_stitch_native_3d = df_concat[(df_concat['dataset'] == 'Dataset902_tumMSStitchedRegion') & (df_concat['model'] == '3D')]
        df_stitch_native_2d = df_concat[(df_concat['dataset'] == 'Dataset902_tumMSStitchedRegion') & (df_concat['model'] == '2D')]
        df_chunks_straight_2d = df_concat[(df_concat['dataset'] == 'Dataset903_tumMSChunksStraightRegion') & (df_concat['model'] == '2D')]
        df_stitch_straight_2d = df_concat[(df_concat['dataset'] == 'Dataset904_tumMSStitchedStraightRegion') & (df_concat['model'] == '2D')]

        # TODO 2. bavaria and polyNYU models (add?)
        # df_chunks_twosites_2d = df_concat[(df_concat['dataset'] == 'Dataset910_tumMSChunksPolyNYUAxialRegion') & (df_concat['model'] == '2D')]

        # ensure that the two dataframes have the same number of rows
        assert len(df_chunks_native_2d) == len(df_stitch_native_2d) == len(df_chunks_straight_2d) == len(df_stitch_straight_2d), \
            f"Number of subjects differ across datasets, paired tests cannot be performed"

        # Combine all dataframes based on 'prediction', 'fold' columns, and the metric column
        df = pd.DataFrame({
            'prediction': df_chunks_native_2d['prediction'].values,  # Assuming 'prediction' is a common column in all dataframes
            'fold': df_chunks_native_2d['fold'].values,  # Assuming 'fold' is a common column in all dataframes
            f'{metric}_chunks_native_2d': df_chunks_native_2d[metric].values,
            f'{metric}_stitch_native_2d': df_stitch_native_2d[metric].values,
            f'{metric}_chunks_straight_2d': df_chunks_straight_2d[metric].values,
            f'{metric}_stitch_straight_2d': df_stitch_straight_2d[metric].values
        })

        # # Drop rows with NaN values
        # df = df.dropna()
        # Print number of subjects
        logger.info(f'{metrics_short[metric]}: Number of subjects: {len(df)}')

        # Compute Kruskal-Wallis H-test
        stat, p = kruskal(df[metric + '_chunks_native_2d'], df[metric + '_stitch_native_2d'],
                            df[metric + '_chunks_straight_2d'], df[metric + '_stitch_straight_2d'])
        logger.info(f'{metrics_short[metric]}: Kruskal-Wallis H-test: formatted p{format_pvalue(p)}, '
                    f'unformatted p={p:0.6f}')

        # Run post-hoc tests between _chunks_native_2d (i.e. best model) and all other methods
        if p < 0.05:
            p_val_dict = {}
            for method in ['stitch_native_2d', 'chunks_straight_2d', 'stitch_straight_2d']:
                stats, p = wilcoxon(df[metric + '_chunks_native_2d'], df[metric + '_' + method], alternative='greater')
                p_val_dict[method] = p
            # Do Bonferroni correction
            # https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
            _, pvals_corrected, _, _ = multipletests(list(p_val_dict.values()), alpha=0.05, method='bonferroni')
            p_val_dict_corrected = dict(zip(list(p_val_dict.keys()), pvals_corrected))
            # Format p-values using format_pvalue function
            p_val_dict_corrected = {k: format_pvalue(v) for k, v in p_val_dict_corrected.items()}
            logger.info(f'{metrics_short[metric]}: Post-hoc tests between chunks_native_2d and all other methods:\n'
                        f'{p_val_dict_corrected}')
        
        logger.info('')


def compute_kruskal_wallis_test_across_tum_poly(df_concat, list_of_metrics, test_site):
    """
    Compute Kruskal-Wallis H-test (non-parametric version of ANOVA)
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
    :param df_concat:
    :param list_of_metrics:
    :return:
    """
    logger.info('')

    # keep only the filenames for reference and prediction
    df_concat['reference'] = df_concat['reference'].apply(find_filename_in_path)
    df_concat['prediction'] = df_concat['prediction'].apply(find_filename_in_path)

    # Loop across sites
    # for site in df_concat['site'].unique():
    # Loop across metrics
    for metric in list_of_metrics:
        
        # create individual dataframes for each model
        df_chunks_single_site_2d = df_concat[(df_concat['dataset'] == 'Dataset901_tumMSChunksRegion') & (df_concat['model'] == '2D')]
        df_chunks_two_sites_2d = df_concat[(df_concat['dataset'] == 'Dataset910_tumMSChunksPolyNYUAxialRegion') & (df_concat['model'] == '2D')]
        df_deepseg_lesion = df_concat[(df_concat['dataset'] == 'DeepSegLesionInference_tumNeuropoly') & (df_concat['model'] == '3D')]

        # ensure that the two dataframes have the same number of rows
        assert len(df_chunks_single_site_2d) == len(df_chunks_two_sites_2d) == len(df_deepseg_lesion), \
            f"Number of subjects differ across datasets, paired tests cannot be performed"

        # run normality test
        stat, p = normaltest(df_chunks_single_site_2d[metric])
        logger.info(f'{metrics_short[metric]}: Normality test for nnUNet2D single-site: formatted p{format_pvalue(p)}, unformatted p={p:0.6f}')
        stat, p = normaltest(df_chunks_two_sites_2d[metric])
        logger.info(f'{metrics_short[metric]}: Normality test for nnUNet2D two-sites: formatted p{format_pvalue(p)}, unformatted p={p:0.6f}')
        stat, p = normaltest(df_deepseg_lesion[metric])
        logger.info(f'{metrics_short[metric]}: Normality test for DeepSegLesionInference: formatted p{format_pvalue(p)}, unformatted p={p:0.6f}')

        # Combine all dataframes based on 'prediction', 'fold' columns, and the metric column
        df = pd.DataFrame({
            'prediction': df_chunks_single_site_2d['prediction'].values,  # Assuming 'prediction' is a common column in all dataframes
            'fold': df_chunks_single_site_2d['fold'].values,  # Assuming 'fold' is a common column in all dataframes
            f'{metric}_chunks_single_site_2d': df_chunks_single_site_2d[metric].values,
            f'{metric}_chunks_two_sites_2d': df_chunks_two_sites_2d[metric].values,
            f'{metric}_deepseg_lesion': df_deepseg_lesion[metric].values
        })

        # Print number of subjects
        logger.info(f'{metrics_short[metric]}; Number of subjects: {len(df)}; Test site: {test_site.upper()}')

        # Compute Kruskal-Wallis H-test
        stat, p = kruskal(df[metric + '_chunks_single_site_2d'], df[metric + '_chunks_two_sites_2d'], df[metric + '_deepseg_lesion'])
        logger.info(f'{metrics_short[metric]}: Kruskal-Wallis H-test: formatted p{format_pvalue(p)}, unformatted p={p:0.6f}')

        # Run post-hoc tests between _chunks_native_2d (i.e. best model) and all other methods
        if p < 0.05:
            p_val_dict = {}
            for method in ['chunks_single_site_2d', 'deepseg_lesion']:
                # NOTE: here, we are computing pair-wise tests between: 
                # (i) two-sites and single-site model, and (ii) two-sites and deepseg_lesion model
                # to conclude whether the two-sites model is significantly better than the other two
                stats, p = wilcoxon(df[metric + '_chunks_two_sites_2d'], df[metric + '_' + method], alternative='greater')
                p_val_dict[method] = p
            # Do Bonferroni correction
            # https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
            _, pvals_corrected, _, _ = multipletests(list(p_val_dict.values()), alpha=0.05, method='bonferroni')
            p_val_dict_corrected = dict(zip(list(p_val_dict.keys()), pvals_corrected))
            # Format p-values using format_pvalue function
            p_val_dict_corrected = {k: format_pvalue(v) for k, v in p_val_dict_corrected.items()}
            logger.info(f'{metrics_short[metric]}: Post-hoc tests between _chunks_two_sites_2d and all other methods for site {test_site.upper()}:\n'
                        f'{p_val_dict_corrected}')
        
        logger.info('')



def main():

    args = get_parser().parse_args()
    path_out = args.o
    if not os.path.exists(path_out):
        os.makedirs(path_out, exist_ok=True)

    num_models_to_compare = len(args.i)
    if num_models_to_compare < 2:
        raise ValueError("Please provide at least two models to compare")
    
    df_mega = pd.DataFrame()
    num_csvs = 0
    for fldr in args.i:

        # TODO: only the lesion metrics for deepseg_lesion, no fold, no 2d/3d
        # if find_dataset_in_path(fldr) != 'deepseg_lesion':

        df_models = pd.DataFrame()
        models_list = sorted([f for f in os.listdir(fldr) if f.endswith('2d') or f.endswith('3d_fullres')])
        if 'Dataset910_tumMSChunksPolyNYUAxialRegion' in fldr and args.test_site == 'muc':
            test_site = 'tum'
        else:
            test_site = args.test_site

        for model in models_list:

            df_folds = pd.DataFrame()
            # find folders using regex
            folds = sorted([f for f in os.listdir(os.path.join(fldr, model)) if re.match(r'fold_\d\b', f)])

            for fld in folds:

                # recursively find anima_metrics_mean_lesion.csv
                files = glob.glob(os.path.join(fldr, model, fld, f'test_{test_site}*', f'metrics_final_{args.pred_type}.csv'))
                # files = glob.glob(os.path.join(fldr, model, fld, f'test_{test_site}*', 'metrics_updated.csv'))
                if not files:
                    # NOTE: then, this is the multi-channel model
                    # files = glob.glob(os.path.join(fldr, model, fld, 'test_*', 'anima_stats', 'anima_metrics_mean.csv'))
                    continue

                print(f"Processing: {files[0].replace('/home/GRAMES.POLYMTL.CA/u114716/nnunet-v2/nnUNet_results', '')}")
                num_csvs += 1

                df = pd.read_csv(files[0])
                df['fold'] = fld
                df['dataset'] = find_dataset_in_path(fldr)
                df['model'] = find_model_in_path(model)

                # NOTE: because multi-channel model has only 1 label, it has to be renamed to 2.0 to match
                # the label id with the region-based models
                if 'multi-channel' in df['dataset'].values[0]:
                    df['label'] = 2.0

                df_folds = pd.concat([df_folds, df])
            
            df_models = pd.concat([df_models, df_folds])

        df_mega = pd.concat([df_mega, df_models])
    
    print(f"Total files: {num_csvs}")
    print(f"Total Rows: {len(df_mega)}")
    
    # remove from stitched datasets
    # these files were causing discrepancies in the total number of subjects between datasets
    exclude_files = [
        'tumMSStitchedRegion_sub-m514993_ses-20230123_acq-ax_T2w',
        'tumMSStitchedRegion_sub-m072952_ses-20190531_acq-ax_T2w',
        'tumMSStitchedStraightRegion_sub-m514993_ses-20230123_acq-ax_T2w',
        'tumMSStitchedStraightRegion_sub-m072952_ses-20190531_acq-ax_T2w',
    ]
    df_mega = df_mega[~df_mega['prediction'].str.contains('|'.join(exclude_files))]


    if args.pred_type == 'lesion':
        metrics_to_plot = {
            'DiceSimilarityCoefficient': 'Dice Score',
            'RelativeVolumeError': 'Relative Volume Error',
            'LesionWiseF1Score': 'Lesion-wise F1 Score',
        }
    elif args.pred_type == 'sc':
        metrics_to_plot = {
            'DiceSimilarityCoefficient': 'Dice Score',
            'RelativeVolumeError': 'Relative Volume Error',
        }
    
    if args.compare_across == 'tum':

        logger.info(f"Total files: {num_csvs}")
        logger.info(f"Total Rows: {len(df_mega)}")

        # generate raincloud plots
        create_rainplot(args, df_mega, metrics_to_plot, path_out, pred_type=args.pred_type)

        # TODO: figure out how to update this for SC segmentation as well
        if args.pred_type == 'lesion':
            # compute statistical tests
            # 1. Wilcoxon signed-rank test between 2D and 3D models within each dataset
            compute_wilcoxon_test(df_mega, metrics_to_plot.keys())
            # 2. Kruskal-Wallis H-test between best-performing models from each dataset
            compute_kruskal_wallis_test_across_tum_datasets(df_mega, metrics_to_plot.keys())
            # 3. Generate regplot for additional metrics
            create_regplot_lesion_count(args, df_mega, path_out)

    elif args.compare_across == 'tum-poly':

        # Add the deepseg_lesion metrics csv manually
        deepseg_lesion_path = f"{os.path.dirname(args.i[0])}/DeepSegLesionInference_tumNeuropoly"
        if args.test_site == 'muc': 
            deepseg_lesion_path = os.path.join(deepseg_lesion_path, "test_tum_deepseg-lesion_stacked/metrics_final_lesion.csv")
        else:
            deepseg_lesion_path = os.path.join(deepseg_lesion_path, f"test_{args.test_site}_deepseg-lesion/metrics_final_lesion.csv")
        logger.info(f"Processing: {deepseg_lesion_path.replace(f'{os.path.dirname(args.i[0])}', '')}")
        df_deepseg = pd.read_csv(deepseg_lesion_path)
        df_deepseg['dataset'] = 'DeepSegLesionInference_tumNeuropoly'
        df_deepseg['model'] = '3D'
        df_deepseg['label'] = 1.0
        df_deepseg['fold'] = 'fold_0'

        df_mega = pd.concat([df_mega, df_deepseg])

        # keep only the rows with fold_0, or else, TUM models have 126*3=378 rows and DeepSegLesion has only 126 rows
        df_mega = df_mega[df_mega['fold'] == 'fold_0']

        logger.info(f"Total files: {num_csvs}")
        logger.info(f"Total Rows: {len(df_mega)}")

        # # remove deepseg_lesion from the dataframe
        # df_mega = df_mega[df_mega['dataset'] != 'DeepSegLesionInference_tumNeuropoly']

        if args.pred_type == 'lesion':
            # generate raincloud plots
            create_rainplot(args, df_mega, metrics_to_plot, path_out, pred_type=args.pred_type)

            # compute statistical tests
            # 1. Kruskal-Wallis H-test between deepseg_lesion, 2D Chunks single-site and two-sites models
            compute_kruskal_wallis_test_across_tum_poly(df_mega, metrics_to_plot.keys(), test_site=args.test_site)


    # save the mega dataframe
    df_mega = df_mega.drop(columns=['label'])
    df_mega = df_mega.round(3)
    df_mega.to_csv(os.path.join(path_out, f'{args.pred_type}_metrics.csv'), index=False)



if __name__ == '__main__':
    main()