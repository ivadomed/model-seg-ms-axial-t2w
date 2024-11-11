"""
This (cool) script generates a LaTeX table with the lesion-wise metrics for the paper.
The input to the script is a CSV file containing the metrics across all the sites and models, which is 
output by the 'plot_metrics_per_site.py' script in the 'model_seg_sci/testing' directory.
The table is printed to the terminal, which can be copied to the .tex file.

"""

import pandas as pd
import argparse


metrics_to_rename = {
    'DiceSimilarityCoefficient_mean': 'Dice',
    'LesionWiseF1Score_mean': 'F1ScoreL',
    'NormalizedSurfaceDistance_mean': 'NSD',
    'DeltaLesionsCount_mean': 'DeltaLesions'
}

models_to_rename = {
    'DeepSegLesionInference_tumNeuropoly': 'DeepSeg\nLesion',
    'Dataset901_tumMSChunksRegion': 'ChunksNative\nSingleSite',
    'Dataset910_tumMSChunksPolyNYUAxialRegion': 'ChunksNative\nTwoSites',
}

def get_parser():
    parser = argparse.ArgumentParser(description='Generate table 1 for the paper')
    parser.add_argument('-i', type=str, nargs='+', required=True,
                        help='Input CSV files for each test site containing all metrics')
    return parser


# Helper function to create rows for each model
def create_rows(model, df, metrics_mean_list, metrics_std_list):
    rows = ""
    len_metrics = len(metrics_mean_list)
    wrapped_model = r"\makecell{" + model.replace('\n', r' \\ ') + r"}"
    for i in range(len_metrics):
        if i == 0:
            rows += f"\\multirow{{{len_metrics}}}{{*}}{{{wrapped_model}}} & {metrics_to_rename[metrics_mean_list[i]]} "
        else:
            rows += f" & {metrics_to_rename[metrics_mean_list[i]]} "
        
        for site in df['site'].unique():
            mean = df.loc[(df['model'] == model) & (df['site'] == site), metrics_mean_list[i]].values[0]
            std = df.loc[(df['model'] == model) & (df['site'] == site), metrics_std_list[i]].values[0]
            rows += f"& {mean:.2f} $\pm$ {std:.2f} "
        rows += "\\\\\n"
    return rows


def main():
    args = get_parser().parse_args()

    # Create LaTeX table
    latex_table = r"""
    \begin{table}[htbp]
        \centering
        \setlength{\tabcolsep}{4pt} % Adjust the length as needed
        \caption{Quantitative comparison of segmentation models on in- and out-of-distribution test sites}
        \resizebox{0.8\textwidth}{!}{%
        \begin{tabular}{llccc}
        \toprule
            \multirow{3}{*}{\textbf{Model}} & \multirow{3}{*}{\textbf{Metric}} & \multicolumn{1}{c}{\multirow{2}{*}{\textbf{\makecell{Test Site \\ (in-distribution)}}}} & \multicolumn{2}{c}{\multirow{2}{*}{\textbf{\makecell{Test Sites \\ (out-of-distribution)}}}} \\
            \\
            \cline{3-5} & &
            \multirow{1}{*}{\textbf{TUM}} & \multirow{1}{*}{\textbf{BWH}} & \multirow{1}{*}{\textbf{UCSF}} \\
            \hline
    """

    metrics_to_include = ['DiceSimilarityCoefficient', 'LesionWiseF1Score', 'NormalizedSurfaceDistance', 'DeltaLesionsCount']

    df_mega = pd.DataFrame()
    # read the csv files
    for i, csv in enumerate(args.i): 
        
        print(f"Reading {csv}")
        df = pd.read_csv(csv)

        # hacky fix -- set the model to 2D where dataset is 'DeepSegLesionInference_tumNeuropoly'
        for dataset in df['dataset'].unique():
            if 'DeepSegLesionInference_tumNeuropoly' in dataset:
                df.loc[df['dataset'] == dataset, 'model'] = '2D'
            else:
                pass

        # remove the rows where models is '3D'
        df = df[df['model'] != '3D']
            
        if df['prediction'].str.contains('sub-m').any():
            df['site'] = 'TUM'
        elif df['prediction'].str.contains('sub-bwh').any():
            df['site'] = 'BWH'
        elif df['prediction'].str.contains('sub-ucsf').any():
            df['site'] = 'UCSF'

        # remove the model column
        df = df.drop(columns=['model', 'reference', 'prediction', 'fold'])

        # rename the dataset column to 'model'
        df = df.rename(columns={'dataset': 'model'})

        # compute the difference in lesion count
        df['DeltaLesionsCount'] = abs(df['PredLesionsCount'] - df['RefLesionsCount'])

        # keep only the metrics in metrics_to_include
        df = df[['site', 'model'] + metrics_to_include]

        # in a new dataframe, compute the mean and std of the metrics for each model and add a suffix _mean and _std
        df_mean = df.groupby(['site', 'model']).mean().add_suffix('_mean').reset_index()
        df_std = df.groupby(['site', 'model']).std().add_suffix('_std').reset_index()

        df_concat = pd.DataFrame()
        # concatenate the mean and std dataframes
        df_concat = pd.concat([df_concat, df_mean, df_std], axis=1)
        # remove the duplicate columns
        df_concat = df_concat.loc[:, ~df_concat.columns.duplicated()]

        # reorder the columns with mean adn std next to each other
        df_concat = df_concat[['model', 'site', \
                               'DiceSimilarityCoefficient_mean', 'DiceSimilarityCoefficient_std', \
                               'NormalizedSurfaceDistance_mean', 'NormalizedSurfaceDistance_std', \
                               'LesionWiseF1Score_mean', 'LesionWiseF1Score_std', \
                                'DeltaLesionsCount_mean', 'DeltaLesionsCount_std']]
        # # move model, site to the front
        # cols = df_concat.columns.tolist()
        # cols = cols[-2:] + cols[:-2]
        # df_concat = df_concat[cols]

        metrics_mean_list = df_concat.columns[df_concat.columns.str.contains('mean')].tolist()
        metrics_std_list = df_concat.columns[df_concat.columns.str.contains('std')].tolist()

        # Concatenate the dataframes
        df_mega = pd.concat([df_mega, df_concat])

    # Rename models
    df_mega['model'] = df_mega['model'].map(models_to_rename)

    # Generate rows for each model 
    for model in df_mega['model'].unique():
        latex_table += create_rows(model, df_mega, metrics_mean_list, metrics_std_list)
        latex_table += "\\hline\n"

    latex_table += r"""
        \bottomrule
        \end{tabular}%
        }
        \label{tab:metrics}
    \end{table}
    """

    print(latex_table)

if __name__ == '__main__':
    main()