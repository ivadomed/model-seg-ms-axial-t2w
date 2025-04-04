"""
This (cool) script generates a LaTeX table with the lesion-wise metrics for the paper.
The input to the script is a CSV file containing the metrics across all the sites and models, which is 
output by the 'plot_metrics_per_site.py' script in the 'model_seg_sci/testing' directory.
The table is printed to the terminal, which can be copied to the .tex file.

The table is formatted with models as columns and test sites as rows.
"""

import pandas as pd
import argparse


metrics_to_rename = {
    'DiceSimilarityCoefficient_mean': 'Dice',
    'LesionWiseF1Score_mean': 'F1ScoreL',
    'LesionWisePositivePredictiveValue_mean': 'PPVL',
    'NormalizedSurfaceDistance_mean': 'NSD',
    'DeltaLesionsCount_mean': 'DeltaLesions'
}

models_to_rename = {
    'DeepSegLesionInference_tumNeuropoly_3D': 'DeepSegLesion',
    'MSLesionAgnosticInference_tumNeuroPoly_3D': 'MSLesionSeg(SCTv6.5)',
    'Dataset901_tumMSChunksRegion_2D': 'ChunksNativeSingleSite',
    # 'Dataset901_tumMSChunksRegion_3D': 'ChunksNative\nSingleSite\n3D',
    'Dataset910_tumMSChunksPolyNYUAxialRegion_2D': 'ChunksNativeTwoSites',
    # 'Dataset910_tumMSChunksPolyNYUAxialRegion_3D': 'ChunksNative\nTwoSites\n3D',
}

def get_parser():
    parser = argparse.ArgumentParser(description='Generate table 1 for the paper')
    parser.add_argument('-i', type=str, nargs='+', required=True,
                        help='Input CSV files for each test site containing all metrics')
    return parser


# Helper function to create rows for each test site and metric
def create_rows(site, df, metrics_mean_list, metrics_std_list, models):
    rows = ""
    len_metrics = len(metrics_mean_list)
    
    for i in range(len_metrics):
        if i == 0:
            rows += f"\\multirow{{{len_metrics}}}{{*}}{{{site}}} & {metrics_to_rename[metrics_mean_list[i]]} "
        else:
            rows += f" & {metrics_to_rename[metrics_mean_list[i]]} "
        
        for model in models:
            if df[(df['model'] == model) & (df['site'] == site)].empty:
                rows += f"& - "
            else:
                mean = df.loc[(df['model'] == model) & (df['site'] == site), metrics_mean_list[i]].values[0]
                std = df.loc[(df['model'] == model) & (df['site'] == site), metrics_std_list[i]].values[0]
                rows += f"& {mean:.2f} $\pm$ {std:.2f} "
        rows += "\\\\\n"
    return rows


def main():
    args = get_parser().parse_args()

    df_mega = pd.DataFrame()
    # read the csv files
    for i, csv in enumerate(args.i): 
        print(f"Reading {csv}")
        df = pd.read_csv(csv)
            
        if df['prediction'].str.contains('sub-m').any():
            df['site'] = 'TUM'
        elif df['prediction'].str.contains('sub-bwh').any():
            df['site'] = 'BWH'
        elif df['prediction'].str.contains('sub-ucsf').any():
            df['site'] = 'UCSF'

        # compute the difference in lesion count
        df['DeltaLesionsCount'] = abs(df['PredLesionsCount'] - df['RefLesionsCount'])

        # merge the dataset column with the model column as dataset_model
        df['dataset_new'] = df['dataset'] + '_' + df['model']
        df = df.drop(columns=['dataset', 'model', 'reference', 'prediction', 'fold'])

        # rename the dataset column to 'model'
        df = df.rename(columns={'dataset_new': 'model'})

        # keep only the metrics we need
        metrics_to_include = ['DiceSimilarityCoefficient', 'LesionWiseF1Score', 'NormalizedSurfaceDistance', 'DeltaLesionsCount', 'LesionWisePositivePredictiveValue']
        df = df[['site', 'model'] + metrics_to_include]

        # compute mean and std for each metric
        df_mean = df.groupby(['site', 'model']).mean().add_suffix('_mean').reset_index()
        df_std = df.groupby(['site', 'model']).std().add_suffix('_std').reset_index()

        df_concat = pd.DataFrame()
        # concatenate the mean and std dataframes
        df_concat = pd.concat([df_concat, df_mean, df_std], axis=1)
        # remove the duplicate columns
        df_concat = df_concat.loc[:, ~df_concat.columns.duplicated()]

        # reorder the columns with mean and std next to each other
        df_concat = df_concat[['model', 'site', 
                               'DiceSimilarityCoefficient_mean', 'DiceSimilarityCoefficient_std',
                               'NormalizedSurfaceDistance_mean', 'NormalizedSurfaceDistance_std',
                               'LesionWisePositivePredictiveValue_mean', 'LesionWisePositivePredictiveValue_std',
                               'LesionWiseF1Score_mean', 'LesionWiseF1Score_std',
                               'DeltaLesionsCount_mean', 'DeltaLesionsCount_std']]

        # Concatenate the dataframes
        df_mega = pd.concat([df_mega, df_concat])

    # Rename models
    df_mega['model'] = df_mega['model'].map(models_to_rename)

    # Get the list of metrics and models
    metrics_mean_list = df_mega.columns[df_mega.columns.str.contains('mean')].tolist()
    metrics_std_list = df_mega.columns[df_mega.columns.str.contains('std')].tolist()
    models = df_mega['model'].unique().tolist()
    sites = ['TUM', 'BWH', 'UCSF']

    # Generate LaTeX table
    model_columns = " & ".join([f"\\multicolumn{{1}}{{c}}{{\\textbf{{{model}}}}} " for model in models])
    # model_columns = " & ".join([f"\\multicolumn{{1}}{{c}}{{\\textbf{{{model.replace(chr(10), ' ')}}}}} " for model in models])
    # print(model_columns)
    # exit()
    
    latex_table = f"""
    \\begin{{table}}[htbp]
        \\centering
        \\setlength{{\\tabcolsep}}{{4pt}} % Adjust the length as needed
        \\caption{{Quantitative comparison of segmentation models on in- and out-of-distribution test sites}}
        \\resizebox{{\\textwidth}}{{!}}{{%
        \\begin{{tabular}}{{ll{"|c" * len(models)}}}
        \\toprule
            \\multirow{{2}}{{*}}{{\\textbf{{Test Site}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Metric}}}} & \\multicolumn{{{len(models)}}}{{c}}{{\\textbf{{Models}}}} \\\\
            \\cline{{3-{2 + len(models)}}}
            & & {model_columns} \\\\
            \\hline
    """

    # Generate rows for each test site
    for site in sites:
        # distribution_type = "(in-distribution)" if site == "TUM" else "(out-of-distribution)"
        sample_size = "{(n=126)}" if site == "TUM" else "{(n=80)}" if site == "BWH" else "{(n=32)}"
        site_label = f"\\textbf{{{site}\n{sample_size}}}" # {distribution_type}"
        
        latex_table += create_rows(site, df_mega, metrics_mean_list, metrics_std_list, models)
        latex_table += "\\hline\n"

    latex_table += """
        \\bottomrule
        \\end{tabular}%
        }
        \\label{tab:metrics}
    \\end{table}
    """

    print(latex_table)

if __name__ == '__main__':
    main()