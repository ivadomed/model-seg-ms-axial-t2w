"""
This script evaluates the reference segmentations and model predictions 
using the "animaSegPerfAnalyzer" command

****************************************************************************************
SegPerfAnalyser (Segmentation Performance Analyzer) provides different marks, metrics 
and scores for segmentation evaluation.
3 categories are available:
    - SEGMENTATION EVALUATION:
        Dice, the mean overlap
        Jaccard, the union overlap
        Sensitivity
        Specificity
        NPV (Negative Predictive Value)
        PPV (Positive Predictive Value)
        RVE (Relative Volume Error) in percentage
    - SURFACE DISTANCE EVALUATION:
        Hausdorff distance
        Contour mean distance
        Average surface distance
    - DETECTION LESIONS EVALUATION:
        PPVL (Positive Predictive Value for Lesions)
        SensL, Lesion detection sensitivity
        F1 Score, a F1 Score between PPVL and SensL

Results are provided as follows: 
Jaccard;    Dice;   Sensitivity;    Specificity;    PPV;    NPV;    RelativeVolumeError;    
HausdorffDistance;  ContourMeanDistance;    SurfaceDistance;  PPVL;   SensL;  F1_score;       

NbTestedLesions;    VolTestedLesions;  --> These metrics are computed for images that 
                                            have no lesions in the GT
****************************************************************************************

Mathematical details on how these metrics are computed can be found here:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6135867/pdf/41598_2018_Article_31911.pdf

and in Section 4 of this paper (for how the subjects with no lesions are handled):
https://portal.fli-iam.irisa.fr/files/2021/06/MS_Challenge_Evaluation_Challengers.pdf

INSTALLATION:
##### STEP 0: Install git lfs via apt if you don't already have it.
##### STEP 1: Install ANIMA #####
cd ~
mkdir anima/
cd anima/
wget -q https://github.com/Inria-Visages/Anima-Public/releases/download/v4.2/Anima-Ubuntu-4.2.zip   (change version to latest)
unzip Anima-Ubuntu-4.2.zip
git lfs install
git clone --depth 1 https://github.com/Inria-Visages/Anima-Scripts-Public.git
git clone --depth 1 https://github.com/Inria-Visages/Anima-Scripts-Data-Public.git

##### STEP 2: Configure directories #####
# Variable names and section titles should stay the same
# Put this file in ${HOME}/.anima/config.txt
# Make the anima variable point to your Anima public build
# Make the extra-data-root point to the data folder of Anima-Scripts
# The last folder separator for each path is crucial, do not forget them
# Use full paths, nothing relative or using tildes

cd ~
mkdir .anima/
touch .anima/config.txt

echo "[anima-scripts]" >> .anima/config.txt
echo "anima = ${HOME}/anima/Anima-Binaries-4.2/" >> .anima/config.txt
echo "anima-scripts-public-root = ${HOME}/anima/Anima-Scripts-Public/" >> .anima/config.txt
echo "extra-data-root = ${HOME}/anima/Anima-Scripts-Data-Public/" >> .anima/config.txt

USAGE:
python compute_anima_metrics.py --pred_folder <path_to_predictions_folder> 
--gt_folder <path_to_gt_folder> -dname <dataset_name> --label-type <sc/lesion>


NOTE 1: For checking all the available options run the following command from your terminal: 
      <anima_binaries_path>/animaSegPerfAnalyzer -h
NOTE 2: We use certain additional arguments below with the following purposes:
      -i -> input image, -r -> reference image, -o -> output folder
      -d -> evaluates surface distance, -l -> evaluates the detection of lesions
      -a -> intra-lesion evalulation (advanced), -s -> segmentation evaluation, 
      -X -> save as XML file  -A -> prints details on output metrics and exits

Authors: Naga Karthik, Jan Valosek
"""

import os
import glob
import pandas as pd
import subprocess
import argparse
from collections import defaultdict
import xml.etree.ElementTree as ET
import numpy as np
import nibabel as nib
from test_utils import fetch_filename_details
from loguru import logger



def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Compute test metrics using animaSegPerfAnalyzer')

    # Arguments for model, data, and training
    parser.add_argument('--pred-folder', required=True, type=str,
                        help='Path to the folder containing nifti images of test predictions')
    parser.add_argument('--gt-folder', required=True, type=str,
                        help='Path to the folder containing nifti images of GT labels')
    parser.add_argument('--training-type', required=True, type=str, default='region-based',
                        help='Type of training. Options: region-based, multi-channel, single-channel')
    # parser.add_argument('--label-type', required=True, type=str, choices=['chunks', 'stitched'],
    #                     help='Type of label. Stitched or Chunks.'
    #                         'if chunks then predictions will be stacked and then the metrics will be computed')
    # parser.add_argument('-o', '--out', required=True, type=str,
    #                     help='name of the csv file to save the metrics')

    return parser


def get_test_metrics_by_dataset(pred_folder, gt_folder, output_folder, anima_binaries_path, training_type):
    """
    Computes the test metrics given folders containing nifti images of test predictions 
    and GT images by running the "animaSegPerfAnalyzer" command
    """
    
    if training_type == "region-based":

        # glob all the predictions and GTs and get the last three digits of the filename
        pred_files = sorted(glob.glob(os.path.join(pred_folder, "*.nii.gz")))
        gt_files = sorted(glob.glob(os.path.join(gt_folder, "*.nii.gz")))

        dataset_name_nnunet = fetch_filename_details(pred_files[0])[0]

        # loop over the predictions and compute the metrics
        for pred_file, gt_file in zip(pred_files, gt_files):
            
            # _, sub_pred, ses_pred, idx_pred, _ = fetch_filename_details(pred_file)
            # _, sub_gt, ses_gt, idx_gt, _ = fetch_filename_details(gt_file)
            _, sub_ses_pred, idx_pred = fetch_filename_details(pred_file)
            _, sub_ses_gt, idx_gt = fetch_filename_details(gt_file)

            # make sure the subject and session IDs match
            # print(f"Subject and session IDs for Preds and GTs: {sub_pred}_{ses_pred}_{idx_pred}, {sub_gt}_{ses_gt}_{idx_gt}")
            assert sub_ses_pred == sub_ses_gt, 'Subject and session IDs for Preds and GTs do not match. Please check the filenames.'
            logger.info("Processing: ")
            logger.info(f"\tPred: {sub_ses_pred}; \tGT: {sub_ses_gt}")
            
            for seg in ['sc', 'lesion']:
                # load the predictions and GTs
                pred_npy = nib.load(pred_file).get_fdata()
                gt_npy = nib.load(gt_file).get_fdata()

                if seg == 'sc':
                    pred_npy = np.array(pred_npy == 1, dtype=float)
                    gt_npy = np.array(gt_npy == 1, dtype=float)                
                
                elif seg == 'lesion':
                    pred_npy = np.array(pred_npy == 2, dtype=float)
                    gt_npy = np.array(gt_npy == 2, dtype=float)
                
                # Save the binarized predictions and GTs
                pred_nib = nib.Nifti1Image(pred_npy, affine=np.eye(4))
                gtc_nib = nib.Nifti1Image(gt_npy, affine=np.eye(4))
                nib.save(img=pred_nib, filename=os.path.join(pred_folder, f"{dataset_name_nnunet}_{sub_ses_pred}_{seg}.nii.gz"))
                nib.save(img=gtc_nib, filename=os.path.join(gt_folder, f"{dataset_name_nnunet}_{sub_ses_gt}_{seg}.nii.gz"))

                # Run ANIMA segmentation performance metrics on the predictions
                if seg == 'sc':
                    seg_perf_analyzer_cmd = '%s -i %s -r %s -o %s -d -s -X'
                elif seg == 'lesion':   # add lesion evaluation metrics with `-l`
                    seg_perf_analyzer_cmd = '%s -i %s -r %s -o %s -d -s -l -X'
                
                os.system(seg_perf_analyzer_cmd %
                            (os.path.join(anima_binaries_path, 'animaSegPerfAnalyzer'),
                            os.path.join(pred_folder, f"{dataset_name_nnunet}_{sub_ses_pred}_{seg}.nii.gz"),
                            os.path.join(gt_folder, f"{dataset_name_nnunet}_{sub_ses_gt}_{seg}.nii.gz"),
                            os.path.join(output_folder, f"{sub_ses_gt}_{seg}")))

                # Delete temporary binarized NIfTI files
                os.remove(os.path.join(pred_folder, f"{dataset_name_nnunet}_{sub_ses_pred}_{seg}.nii.gz"))
                os.remove(os.path.join(gt_folder, f"{dataset_name_nnunet}_{sub_ses_gt}_{seg}.nii.gz"))

        # Get all XML filepaths where ANIMA performance metrics are saved for each hold-out subject
        subject_sc_filepaths = [os.path.join(output_folder, f) for f in
                                os.listdir(output_folder) if f.endswith('.xml') and 'sc' in f]
        subject_lesion_filepaths = [os.path.join(output_folder, f) for f in
                                os.listdir(output_folder) if f.endswith('.xml') and 'lesion' in f]
        
        return subject_sc_filepaths, subject_lesion_filepaths

    elif training_type in ["multi-channel", "single-channel"]:
        logger.info("Training Type:", training_type)
        # glob all the predictions and GTs and get the last three digits of the filename
        pred_files = sorted(glob.glob(os.path.join(pred_folder, "*.nii.gz")))
        gt_files = sorted(glob.glob(os.path.join(gt_folder, "*.nii.gz")))

        dataset_name_nnunet = fetch_filename_details(pred_files[0])[0]

        # loop over the predictions and compute the metrics
        for pred_file, gt_file in zip(pred_files, gt_files):
            
            _, sub_ses_pred, idx_pred = fetch_filename_details(pred_file)
            _, sub_ses_gt, idx_gt = fetch_filename_details(gt_file)

            # make sure the subject and session IDs match
            # print(f"Subject and session IDs for Preds and GTs: {sub_pred}_{ses_pred}_{idx_pred}, {sub_gt}_{ses_gt}_{idx_gt}")
            assert sub_ses_pred == sub_ses_gt, 'Subject and session IDs for Preds and GTs do not match. Please check the filenames.'
            print("Processing: ")
            print(f"\tPred: {sub_ses_pred}; \tGT: {sub_ses_gt}")
            
            # load the predictions and GTs
            pred_npy = nib.load(pred_file).get_fdata()
            gt_npy = nib.load(gt_file).get_fdata()
            
            # make sure the predictions are binary because ANIMA accepts binarized inputs only
            pred_npy = np.array(pred_npy > 0.5, dtype=float)
            gt_npy = np.array(gt_npy > 0.5, dtype=float)

            # Save the binarized predictions and GTs
            pred_nib = nib.Nifti1Image(pred_npy, affine=np.eye(4))
            gtc_nib = nib.Nifti1Image(gt_npy, affine=np.eye(4))
            nib.save(img=pred_nib, filename=os.path.join(pred_folder, f"{dataset_name_nnunet}_{sub_ses_pred}.nii.gz"))
            nib.save(img=gtc_nib, filename=os.path.join(gt_folder, f"{dataset_name_nnunet}_{sub_ses_gt}.nii.gz"))

            # Run ANIMA segmentation performance metrics on the predictions            
            # if label_type == 'lesion':
            seg_perf_analyzer_cmd = '%s -i %s -r %s -o %s -d -l -s -X'
            # elif label_type == 'sc':
            #     seg_perf_analyzer_cmd = '%s -i %s -r %s -o %s -d -s -X'
            # else:
            #     raise ValueError('Please specify a valid label type: lesion or sc')

            os.system(seg_perf_analyzer_cmd %
                        (os.path.join(anima_binaries_path, 'animaSegPerfAnalyzer'),
                        os.path.join(pred_folder, f"{dataset_name_nnunet}_{sub_ses_pred}.nii.gz"),
                        os.path.join(gt_folder, f"{dataset_name_nnunet}_{sub_ses_gt}.nii.gz"),
                        os.path.join(output_folder, f"{sub_ses_gt}")))

            # Delete temporary binarized NIfTI files
            os.remove(os.path.join(pred_folder, f"{dataset_name_nnunet}_{sub_ses_pred}.nii.gz"))
            os.remove(os.path.join(gt_folder, f"{dataset_name_nnunet}_{sub_ses_gt}.nii.gz"))

        # Get all XML filepaths where ANIMA performance metrics are saved for each hold-out subject
        subject_filepaths = [os.path.join(output_folder, f) for f in
                                os.listdir(output_folder) if f.endswith('.xml')]
        
        return subject_filepaths


def main():

    parser = get_parser()
    args = parser.parse_args()

    # Dump log file there
    fname_log = f'log_anima_stats.txt'

    # define variables
    pred_folder, gt_folder = args.pred_folder, args.gt_folder
    training_type = args.training_type

    output_folder = os.path.join(pred_folder, f"anima_stats")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    logger.add(output_folder + '/' + fname_log, rotation="10 MB", level="INFO")
    logger.info(f"Saving ANIMA performance metrics to {output_folder}")

    # get the ANIMA binaries path
    cmd = r'''grep "^anima = " ~/.anima/config.txt | sed "s/.* = //"'''
    anima_binaries_path = subprocess.check_output(cmd, shell=True).decode('utf-8').strip('\n')
    logger.info('ANIMA Binaries Path:', anima_binaries_path)
    # version = subprocess.check_output(anima_binaries_path + 'animaSegPerfAnalyzer --version', shell=True).decode('utf-8').strip('\n')
    logger.info('Running ANIMA version:',
          subprocess.check_output(anima_binaries_path + 'animaSegPerfAnalyzer --version', shell=True).decode(
              'utf-8').strip('\n'))

    num_subjects_skipped = 0

    if training_type in ["multi-channel", "single-channel"]:

        # Get all XML filepaths where ANIMA performance metrics are saved for each hold-out subject
        subject_filepaths = get_test_metrics_by_dataset(pred_folder, gt_folder, output_folder, anima_binaries_path,
                                                        training_type=training_type)

        test_metrics = defaultdict(list)
        subject_files_final = []

        # Update the test metrics dictionary by iterating over all subjects
        for subject_filepath in subject_filepaths:
            # subject = os.path.split(subject_filepath)[-1].split('_')[0]
            root_node = ET.parse(source=subject_filepath).getroot()

            # if GT is empty then metrics aren't calculated, hence the only entries in the XML file 
            # NbTestedLesions and VolTestedLesions, both of which are zero. Hence, we can skip subjects
            # with empty GTs by checked if the length of the .xml file is 2
            if len(root_node) == 2:
                logger.info(f"Skipping Subject={os.path.split(subject_filepath)[-1]} ENTIRELY Due to Empty GT!")
                num_subjects_skipped += 1
                continue

            test_metrics['Label'] = 1.0  # corresponds to lesion 

            for metric in list(root_node):
                name, value = metric.get('name'), float(metric.text)

                # if np.isinf(value) or np.isnan(value):
                #     logger.info(f'Skipping Metric={name} for Subject={os.path.split(subject_filepath)[-1]} Due to INF or NaNs!')
                #     continue

                test_metrics[name].append(value)
            
            subject_files_final.append(subject_filepath)

        logger.info(f"Number of Subjects Skipped Due to Empty GTs: {num_subjects_skipped}")

        # convert test_metrics to a dataframe
        df = pd.DataFrame(test_metrics)
        # create a column Prediction and add the sub_ses to it
        df['Prediction'] = [f"{os.path.split(subject_file)[-1]}" for subject_file in subject_files_final]
        # bring the `Prediction` and `Label` columns to the front
        df = df[['Prediction', 'Label'] + [col for col in df.columns if col not in ['Prediction', 'Label']]]
        # sort the dataframe by the `Prediction` column
        df = df.sort_values(by='Prediction')
        # drop any rows iwth NaN values
        df = df.dropna()
        # format output up to 3 decimal places
        df = df.round(3)
        # save the dataframe to a csv file
        df.to_csv(os.path.join(output_folder, f'anima_metrics.csv'), index=False)

        df_mean = (df.drop(columns=['Prediction']).groupby('Label').agg(['mean', 'std']).reset_index())
        # Convert multi-index to flat index
        df_mean.columns = ['_'.join(col).strip() for col in df_mean.columns.values]
        # Rename column `label_` back to `label`
        df_mean.rename(columns={'Label_': 'Label'}, inplace=True)

        df_mean = df_mean.round(3)
        
        # save the dataframe to a csv file
        df_mean.to_csv(os.path.join(output_folder, f'anima_metrics_mean.csv'), index=False)
        
    else:

        # Get all XML filepaths where ANIMA performance metrics are saved for each hold-out subject
        subject_sc_filepaths, subject_lesion_filepaths = \
            get_test_metrics_by_dataset(pred_folder, gt_folder, output_folder, anima_binaries_path,
                                        training_type=training_type)

        # loop through the sc and lesion filepaths and get the metrics
        for subject_filepaths in [subject_sc_filepaths, subject_lesion_filepaths]:
        
            test_metrics = defaultdict(list)
            subject_files_final = []

            # Update the test metrics dictionary by iterating over all subjects
            for subject_filepath in subject_filepaths:
                
                seg_type = os.path.split(subject_filepath)[-1].split('_')[-2]
                root_node = ET.parse(source=subject_filepath).getroot()

                # if GT is empty then metrics aren't calculated, hence the only entries in the XML file 
                # NbTestedLesions and VolTestedLesions, both of which are zero. Hence, we can skip subjects
                # with empty GTs by checked if the length of the .xml file is 2
                if len(root_node) == 2:
                    logger.info(f"Skipping Subject={os.path.split(subject_filepath)[-1]} ENTIRELY Due to Empty GT!")
                    num_subjects_skipped += 1
                    continue

                test_metrics['Label'] = 1.0 if seg_type == 'sc' else 2.0

                for metric in list(root_node):
                    name, value = metric.get('name'), float(metric.text)

                    # if np.isinf(value) or np.isnan(value):
                    #     logger.info(f'Skipping Metric={name} for Subject={os.path.split(subject_filepath)[-1]} Due to INF or NaNs!')
                    #     continue

                    test_metrics[name].append(value)
                
                subject_files_final.append(subject_filepath)

            # convert test_metrics to a dataframe
            df = pd.DataFrame(test_metrics)
            # create a column Prediction and add the sub_ses to it
            df['Prediction'] = [f"{os.path.split(subject_file)[-1]}" for subject_file in subject_files_final]
            # bring the `Prediction` and `Label` columns to the front
            df = df[['Prediction', 'Label'] + [col for col in df.columns if col not in ['Prediction', 'Label']]]
            # sort the dataframe by the `Prediction` column
            df = df.sort_values(by='Prediction')
            # drop any rows iwth NaN values
            df = df.dropna()
            # format output up to 3 decimal places
            df = df.round(3)
            # save the dataframe to a csv file
            df.to_csv(os.path.join(output_folder, f'anima_metrics_{seg_type}.csv'), index=False)

            df_mean = (df.drop(columns=['Prediction']).groupby('Label').agg(['mean', 'std']).reset_index())
            # Convert multi-index to flat index
            df_mean.columns = ['_'.join(col).strip() for col in df_mean.columns.values]
            # Rename column `label_` back to `label`
            df_mean.rename(columns={'Label_': 'Label'}, inplace=True)

            df_mean = df_mean.round(3)
            
            # save the dataframe to a csv file
            df_mean.to_csv(os.path.join(output_folder, f'anima_metrics_mean_{seg_type}.csv'), index=False)


if __name__ == '__main__':
    main()
