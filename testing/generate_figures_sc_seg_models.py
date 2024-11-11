import os
import re
import argparse
from loguru import logger
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import ptitprince as pt
from scipy.stats import wilcoxon, normaltest, kruskal
import scikit_posthocs as sp

LABEL_FONT_SIZE = 15
TICK_FONT_SIZE = 13
PALETTE = ['red', 'blue', 'green']

METHODS_TO_LABEL_SC = {
    'propseg': 'sct_propseg',
    'deepseg_3d': 'sct_deepseg_sc\n3D',
    'deepseg_2d': 'sct_deepseg_sc\n2D',
    # 'monai_v2.0': 'contrast-agnostic\nv2.0',
    'monai_v2.6': 'contrast-agnostic\n(3D)', #\nv2.5',
    'nnunet_2d': 'ChunksNative\n2D',
    'nnunet_3d_fullres': 'StitchedNative\n3D',
    }

metrics_short = {
    'DiceSimilarityCoefficient': 'Dice',
    'RelativeVolumeError': 'RVE',
    'LesionWiseF1Score': 'LesionF1',
}


def get_parser():

    parser = argparse.ArgumentParser(description='Plot metrics per site')
    parser.add_argument('-i', type=str, required=True, 
                        help='Path to the folders containing all model to compare')
    parser.add_argument('-o', type=str, required=True,
                        help='Path to the output folder where the plots will be saved')
    # parser.add_argument('-test-site', type=str, required=True,
    #                     help='Results to compare from which site: `muc` (TUM MS) or `deepseg`')
    parser.add_argument('-pred-type', type=str, default='sc',
                        help='Type of prediction to create plots for: `sc` (spinal cord segmentation) or `lesion`')

    return parser


def find_chunk_in_path(path):
    match = re.search(r'chunk-1|chunk-2|chunk-3', path)
    if match:
        return match.group(0)
    else:
        return 'Unknown'


def find_filename_in_path(path):
    fname = path.split('/')[-1]
    fname = fname.split('.')[0]
    sub, ses = fname.split('_')[0], fname.split('_')[1]
    return f'{sub}_{ses}'


def create_rainplot(args, df, metrics, path_figures):
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

    for metric in metrics.keys():
        fig_size = (11, 6) # if pred_type == 'sc' else (9, 5.5)
        fig, ax = plt.subplots(figsize=fig_size)
        ax = pt.RainCloud(data=df,
                          x='model',
                          y=metric,
                          hue='chunk',
                          palette=PALETTE,
                          order=METHODS_TO_LABEL_SC.keys(), #if pred_type == 'sc' else METHODS_TO_LABEL_LESION.keys(),
                          dodge=True,       # move boxplots next to each other
                          linewidth=0,      # violionplot border line (0 - no line)
                          width_viol=.5,    # violionplot width
                          width_box=.45,     # boxplot width
                          rain_alpha=.7,    # individual points transparency
                          rain_s=2,         # individual points size
                          alpha=.7,         # violin plot transparency
                          box_showmeans=True,  # show mean value inside the boxplots
                          box_meanprops={'marker': '^', 'markerfacecolor': 'black', 'markeredgecolor': 'black',
                                         'markersize': '6'},
                          hue_order=['chunk-1', 'chunk-2', 'chunk-3'],
                          )

        # TODO: include mean +- std for each boxplot above the mean value

        # Change boxplot opacity (.0 means transparent)
        # https://github.com/mwaskom/seaborn/issues/979#issuecomment-1144615001
        for patch in ax.patches:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .0))

        # Include number of subjects for each site into the legend
        handles, labels = ax.get_legend_handles_labels()
        # for i, label in enumerate(labels):
        #     n = len(df[(df['model'] == list(METHODS_TO_LABEL_SC.keys())[0]) & (df['chunk'] == label)])
        #     labels[i] = f'{label}' + ' ($\it{n}$' + f' = {n})'
        # Since the figure contains violionplot + boxplot + scatterplot we are keeping only last two legend entries
        handles, labels = handles[-3:], labels[-3:]
        location = 'upper right' if metric == 'RelativeVolumeError' else 'lower right'
        ax.legend(handles, labels, fontsize=TICK_FONT_SIZE, loc=location)

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
        ax.set_xticklabels(METHODS_TO_LABEL_SC.values(), #if pred_type == 'sc' else METHODS_TO_LABEL_LESION.values(),
                           fontsize=TICK_FONT_SIZE)
        # Increase y-axis label font size
        ax.set_ylabel(metric, fontsize=TICK_FONT_SIZE)
        # Increase y-ticks font size
        ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)

        # Set title
        if args.pred_type == 'sc':
            ax.set_title(f'Test {metrics[metric]} for Spinal Cord Segmentation', fontsize=LABEL_FONT_SIZE)
        else:
            ax.set_title(f'Test {metrics[metric]} for Lesion Segmentation', fontsize=LABEL_FONT_SIZE)

        # Move grid to background (i.e. behind other elements)
        ax.set_axisbelow(True)
        # Add horizontal grid lines and change its opacity
        ax.yaxis.grid(True, alpha=0.3)
        # modify the y-axis ticks
        if args.pred_type == 'lesion' and metric != 'RelativeVolumeError':
            ax.set_yticks(np.arange(0, 1.1, 0.1))
        elif args.pred_type == 'sc' and metric != 'RelativeVolumeError':
            # ax.set_yticks(np.arange(0.55, 1.025, 0.05))
            ax.set_ylim(0.55, 1.025)
            # ax.set_yticks(np.arange(0.6, 1.025, 0.05))
        elif metric == 'RelativeVolumeError':
            ax.set_ylim(-50, 95)
        
        plt.tight_layout()

        # save figure
        fname_fig = os.path.join(path_figures, f'rainplot_{args.pred_type}_{metric}.png')
        plt.savefig(fname_fig, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f'Created: {fname_fig}')


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


def compute_kruskal_wallis_test(df_concat, list_of_metrics):
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

    # Loop across metrics
    for metric in list_of_metrics:

        for chunk in ["chunk-1", "chunk-2", "chunk-3"]:
        
            # create individual dataframes for each model
            df_chunks_native = df_concat[(df_concat['model'] == 'nnunet_2d') & (df_concat['chunk'] == chunk)]
            df_stitched_native = df_concat[(df_concat['model'] == 'nnunet_3d_fullres') & (df_concat['chunk'] == chunk)]
            df_contrast_agnostic = df_concat[(df_concat['model'] == 'monai_v2.6') & (df_concat['chunk'] == chunk)]
            df_deepseg_2d = df_concat[(df_concat['model'] == 'deepseg_2d') & (df_concat['chunk'] == chunk)]
            df_deepseg_3d = df_concat[(df_concat['model'] == 'deepseg_3d') & (df_concat['chunk'] == chunk)]
            df_propseg = df_concat[(df_concat['model'] == 'propseg') & (df_concat['chunk'] == chunk)]

            # NOTE: contrast-agnostic and stitched models have 5 more images per each chunk i.e. 15 more images
            # so excluding them to match the number of subjects
            # from stitched, exclude prediction files that are not in chunks
            df_stitched_native = df_stitched_native[df_stitched_native['prediction'].isin(df_chunks_native['prediction'])]
            df_contrast_agnostic = df_contrast_agnostic[df_contrast_agnostic['prediction'].isin(df_chunks_native['prediction'])]

            # ensure that the two dataframes have the same number of rows
            assert len(df_chunks_native) == len(df_stitched_native) == len(df_contrast_agnostic) == len(df_deepseg_2d) == len(df_deepseg_3d) == len(df_propseg), \
                f"Number of subjects differ across datasets, paired tests cannot be performed"

            methods_to_compare = ['nnunet_2d', 'nnunet_3d_fullres', 'monai_v2.6', 'deepseg_2d', 'deepseg_3d', 'propseg']
            # run normality tests
            for method in methods_to_compare:
                stat, p = normaltest(df_concat[metric])
                logger.info(f'{metrics_short[metric]}: Normality test for {method}: formatted p{format_pvalue(p)}, unformatted p={p:0.6f}')

            # Combine all dataframes based on 'prediction', 'fold' columns, and the metric column
            df = pd.DataFrame({
                'prediction': df_chunks_native['prediction'].values,  # Assuming 'prediction' is a common column in all dataframes
                'chunk': df_chunks_native['chunk'].values,
                f'{metric}_chunks_native': df_chunks_native[metric].values,
                f'{metric}_stitched_native': df_stitched_native[metric].values,
                f'{metric}_contrast_agnostic': df_contrast_agnostic[metric].values,
                f'{metric}_deepseg_2d': df_deepseg_2d[metric].values,
                f'{metric}_deepseg_3d': df_deepseg_3d[metric].values,
                f'{metric}_propseg': df_propseg[metric].values
            })

            # Print number of subjects
            logger.info(f'{metrics_short[metric]}; Number of subjects: {len(df)}; Test site: TUM; chunk: {chunk}')

            # Compute Kruskal-Wallis H-test between 5 groups
            stat, p = kruskal(df[f'{metric}_chunks_native'], df[f'{metric}_stitched_native'], df[f'{metric}_contrast_agnostic'],
                df[f'{metric}_deepseg_2d'], df[f'{metric}_deepseg_3d'], df[f'{metric}_propseg'])
            
            logger.info(f'{metrics_short[metric]}: Kruskal-Wallis H-test: formatted p{format_pvalue(p)}, unformatted p={p:0.6f}')

            if p < 0.05:
                # Perform Dunn's test as post-hoc analysis (https://www.geeksforgeeks.org/how-to-perform-dunns-test-in-python/)            
                
                # reframe the df to have all methods in a column and the metric values in another column
                df_temp = pd.melt(df, id_vars=['prediction', 'chunk'], value_vars=[f'{metric}_chunks_native', f'{metric}_stitched_native',
                    f'{metric}_contrast_agnostic', f'{metric}_deepseg_2d', f'{metric}_deepseg_3d', f'{metric}_propseg'],
                    var_name='model', value_name='metric')

                # val_col here is the "dependent" variable i.e. the metric scores and the group_col is the "independent" variable i.e. the 
                # model which is used to get the metric scores
                p_post_hoc = sp.posthoc_dunn(df_temp, p_adjust='holm', group_col='model', val_col='metric')

                # check if p_post_hoc is symmetric
                np.testing.assert_array_equal(p_post_hoc.values, p_post_hoc.values.T), "p_post_hoc matrix is not symmetric"
                
                # p_post_hoc is a len(model) x len(model) matrix with p-values for each pair of models and it is symmetric
                # so we only need to print the upper triangle of the matrix
                logger.info(f"{metrics_short[metric]}: Post-hoc Dunn's test; Site: TUM; chunk: {chunk}")
                for i in range(p_post_hoc.shape[0]):
                    for j in range(p_post_hoc.shape[1]):
                        if i < j:
                            model_a = p_post_hoc.index[i].replace(metric, '')
                            model_b = p_post_hoc.columns[j].replace(metric, '')
                            p_formatted = format_pvalue(p_post_hoc.iloc[i, j])
                            logger.info(f"\t{model_a} vs {model_b}: formatted p{p_formatted}, unformatted p={p_post_hoc.iloc[i, j]:0.6f}")
            
            logger.info('')



def main():

    args = get_parser().parse_args()
    path_out = args.o

    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_out = f"{path_out}_{date_time}"
    logger.add(os.path.join(path_out, 'log.txt'), rotation='10 MB', level='INFO')

    models_to_compare = os.listdir(args.i)
    
    df_mega = pd.DataFrame()
    num_csvs = 0
    for model in models_to_compare:

        df_chunks = pd.DataFrame()
        chunks = os.listdir(os.path.join(args.i, model))
        # chunks = [fldr for fldr in os.listdir(os.path.join(args.i, model)) if re.match(r'chunk-\d\b', fldr)]

        for chunk in chunks:
            # recursively find anima_metrics_mean_lesion.csv
            files = glob.glob(os.path.join(args.i, model, chunk, f'metrics_final_{model}_{chunk}_sc.csv'))                
            # files = glob.glob(os.path.join(fldr, model, fld, f'test_{test_site}*', 'metrics_updated.csv'))
            if not files:
                # NOTE: then, this is the multi-channel model
                # files = glob.glob(os.path.join(fldr, model, fld, 'test_*', 'anima_stats', 'anima_metrics_mean.csv'))
                continue

            logger.info(f"Processing: {files[0].replace(f'{os.path.dirname(args.i)}', '')}")
            num_csvs += 1

            df = pd.read_csv(files[0])
            df['chunk'] = chunk
            df['model'] = model
            # drop duplicates
            df = df.drop_duplicates(subset=['prediction'], keep='first')#, 'fold', 'model', 'chunk'], keep='first')

            df_chunks = pd.concat([df_chunks, df])

        df_mega = pd.concat([df_mega, df_chunks])
        # drop the label column
        df_mega = df_mega.drop(columns=['label'])

    # remove the monai_v2.0 model
    df_mega = df_mega[df_mega['model'] != 'monai_v2.0']
    df_mega = df_mega[df_mega['model'] != 'monai_v2.5']
    # df_mega = df_mega.drop_duplicates(subset=['prediction'])#, 'fold', 'model', 'chunk'], keep='first')    
        
    # # remove from stitched datasets
    # # these files were causing discrepancies in the total number of subjects between datasets
    # exclude_files = [
    #     'tumMSStitchedRegion_sub-m514993_ses-20230123_acq-ax_T2w',
    #     'tumMSStitchedRegion_sub-m072952_ses-20190531_acq-ax_T2w',
    #     'tumMSStitchedStraightRegion_sub-m514993_ses-20230123_acq-ax_T2w',
    #     'tumMSStitchedStraightRegion_sub-m072952_ses-20190531_acq-ax_T2w',
    # ]
    # df_mega = df_mega[~df_mega['prediction'].str.contains('|'.join(exclude_files))]

    metrics_to_plot = {
        'DiceSimilarityCoefficient': 'Dice Score',
        'RelativeVolumeError': 'Relative Volume Error',
    }

    logger.info(f"Total files: {num_csvs}")
    logger.info(f"Total Rows: {len(df_mega)}")

    # generate raincloud plots
    create_rainplot(args, df_mega, metrics_to_plot, path_out)

    # perform Kruskal-Wallis H-test and post-hoc Dunn's test
    compute_kruskal_wallis_test(df_mega, {'DiceSimilarityCoefficient': 'Dice Score'})

    # save the mega dataframe
    df_mega = df_mega.round(3)
    df_mega.to_csv(os.path.join(path_out, f'{args.pred_type}_metrics.csv'), index=False)



if __name__ == '__main__':
    main()