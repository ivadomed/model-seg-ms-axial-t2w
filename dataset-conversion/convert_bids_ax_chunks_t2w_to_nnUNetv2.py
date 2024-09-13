"""
Convert BIDS-structured Axial T2w chunk MS lesion images to the nnUNetv2 REGION-BASED training format
depending on the input arguments.

Full details about the format can be found here:
https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

The script to be used on a single dataset or multiple datasets.

Note: the script performs RPI reorientation of the images and labels

Usage example single dataset:
    python convert_bids_to_nnUNetv2.py
        --path-data <path/to/dataset>
        --path-out ${nnUNet_raw}
        -dname bavariaMSChunks
        -dnum 601
        --split 0.8 0.2
        --seed 42
        --region-based
        --exclude <path/to/exclude.yml>

Authors: Julian McGinnis, Naga Karthik, Jan Valosek, Pierre-Louis Benveniste
"""

import argparse
import subprocess
from pathlib import Path
import json
import os
import re
import shutil
import yaml
from collections import OrderedDict
from loguru import logger
from sklearn.model_selection import train_test_split
from utils import binarize_label, create_region_based_label, get_git_branch_and_commit, Image
from tqdm import tqdm
import nibabel as nib


LABEL_SUFFIXES = {
    "tum": ["seg-manual", "lesion-manual"],
    "testing-large": ["seg-manual", "lesion-manual"],
}


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert BIDS-structured dataset to nnUNetV2 REGION-BASED format.')
    parser.add_argument('--path-data', nargs='+', required=True, type=str,
                        help='Path to BIDS dataset(s) (list).')
    parser.add_argument('--path-json', type=str, default=None,
                        help='Path to the JSON file containing NeuroPoly and TUM Axial T2w images.')
    parser.add_argument('--path-out', help='Path to output directory.', required=True)
    parser.add_argument('--dataset-name', '-dname', default='DCMlesionsRegionBased', type=str,
                        help='Specify the task name.')
    parser.add_argument('--dataset-number', '-dnum', default=601, type=int,
                        help='Specify the task number, has to be greater than 500 but less than 999. e.g 502')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to be used for the random number generator split into training and test sets.')
    parser.add_argument('--region-based', action='store_true', default=False,
                        help='If set, the script will create labels for region-based nnUNet training. Default: True')
    # argument that accepts a list of floats as train val test splits
    parser.add_argument('--split', nargs='+', type=float, default=[0.8, 0.2],
                        help='Ratios of training (includes validation) and test splits lying between 0-1. Example: '
                             '--split 0.8 0.2')
    parser.add_argument('--exclude', type=str, default=None,
                        help='YAML file containing the list of subjects to exclude from the dataset.')
    return parser


def get_region_based_label(subject_label_file, subject_image_file, site_name, sub_ses_name, thr=0.5):
    # define path for sc seg file
    subject_seg_file = subject_label_file.replace(f'_{LABEL_SUFFIXES[site_name][1]}', f'_{LABEL_SUFFIXES[site_name][0]}')

    # check if the seg file exists
    if not os.path.exists(subject_seg_file):
        logger.info(f"Spinal cord segmentation file for subject {sub_ses_name} does not exist. Skipping.")
        return None

    # create region-based label
    seg_lesion_nii = create_region_based_label(subject_label_file, subject_seg_file, subject_image_file,
                                               sub_ses_name, thr=thr)

    # save the region-based label
    site_suffix = 'native' if site_name == 'muc' else site_name
    combined_seg_file = subject_label_file.replace(f'_{LABEL_SUFFIXES[site_name][1]}', f'_sc-lesion_{site_suffix}')
    nib.save(seg_lesion_nii, combined_seg_file)

    return combined_seg_file


def create_directories(path_out, site):
    """Create test directories for a specified site.

    Args:
    path_out (str): Base output directory.
    site (str): Site identifier, such as 'dcm-zurich-lesions
    """
    paths = [Path(path_out, f'imagesTs_{site}'),
             Path(path_out, f'labelsTs_{site}'),]

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def fetch_subject_nifti_details(filename_path):
    """
    Get subject ID, session ID and filename from the input BIDS-compatible filename or file path
    The function works both on absolute file path as well as filename
    :param filename_path: input nifti filename (e.g., sub-001_ses-01_T1w.nii.gz) or file path
    (e.g., /home/user/MRI/bids/derivatives/labels/sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz
    :return: subject_session: subject ID and session ID (e.g., sub-001_ses-01) or subject ID (e.g., sub-001)
    Taken from: 
    """

    subject = re.search('sub-(.*?)[_/]', filename_path)     # [_/] means either underscore or slash
    subjectID = subject.group(0)[:-1] if subject else ""    # [:-1] removes the last underscore or slash

    session = re.search('ses-(.*?)[_/]', filename_path)     # [_/] means either underscore or slash
    sessionID = session.group(0)[:-1] if session else ""    # [:-1] removes the last underscore or slash

    orientation = re.search('acq-(.*?)[_/]', filename_path)     # [_/] means either underscore or slash
    orientationID = orientation.group(0)[:-1] if orientation else ""    # [:-1] removes the last underscore or slash

    # search only for axial T2w images
    contrast_pattern =  r'.*_(acq-ax_T2w).*'
    contrast = re.search(contrast_pattern, filename_path)
    contrastID = contrast.group(1) if contrast else ""

    # REGEX explanation
    # . - match any character (except newline)
    # *? - match the previous element as few times as possible (zero or more times)

    # subject_session = subjectID + '_' + sessionID if subjectID and sessionID else subjectID
    return subjectID, sessionID, orientationID, contrastID


def create_yaml(train_niftis, test_nifitis, path_out, args, train_ctr, test_ctr, dataset_commits):
    # create a yaml file containing the list of training and test niftis
    niftis_dict = {
        f"train": sorted(train_niftis),
        f"test": sorted(test_nifitis)
    }

    # write the train and test niftis to a yaml file
    with open(os.path.join(path_out, f"train_test_split_seed{args.seed}.yaml"), "w") as outfile:
        yaml.dump(niftis_dict, outfile, default_flow_style=False)

    # c.f. dataset json generation
    # In nnUNet V2, dataset.json file has become much shorter. The description of the fields and changes
    # can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md#datasetjson
    # this file can be automatically generated using the following code here:
    # https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/dataset_conversion/generate_dataset_json.py
    # example: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task055_SegTHOR.py

    json_dict = OrderedDict()
    json_dict['name'] = args.dataset_name
    json_dict['description'] = args.dataset_name
    json_dict['reference'] = "TBD"
    json_dict['licence'] = "TBD"
    json_dict['release'] = "0.0"
    json_dict['numTraining'] = train_ctr
    json_dict['numTest'] = test_ctr
    json_dict['seed_used'] = args.seed
    json_dict['dataset_versions'] = dataset_commits
    json_dict['image_orientation'] = "RPI"

    # The following keys are the most important ones.
    if args.region_based:
        json_dict['channel_names'] = {
            0: "T2w",
        }

        json_dict['labels'] = {
            "background": 0,
            "sc": [1, 2],
            "lesion": 2,
        }
        json_dict['regions_class_order'] = [1, 2]

    else:
        json_dict['channel_names'] = {
            0: "acq-ax_T2w",
        }

        json_dict['labels'] = {
            "background": 0,
            "lesion": 1,
        }

    # Needed for finding the files correctly. IMPORTANT! File endings must match between images and segmentations!
    json_dict['file_ending'] = ".nii.gz"

    # create dataset_description.json
    json_object = json.dumps(json_dict, indent=4)
    # write to dataset description
    # nn-unet requires it to be "dataset.json"
    dataset_dict_name = f"dataset.json"
    with open(os.path.join(path_out, dataset_dict_name), "w") as outfile:
        outfile.write(json_object)


def main():
    parser = get_parser()
    args = parser.parse_args()

    train_ratio, test_ratio = args.split
    path_out = Path(os.path.join(os.path.abspath(args.path_out), f'Dataset{args.dataset_number}_{args.dataset_name}'))

    # create individual directories for train and test images and labels
    path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
    path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
    # create the training directories
    Path(path_out).mkdir(parents=True, exist_ok=True)
    Path(path_out_imagesTr).mkdir(parents=True, exist_ok=True)
    Path(path_out_labelsTr).mkdir(parents=True, exist_ok=True)

    # save output to a log file
    logger.add(os.path.join(path_out, "logs.txt"), rotation="10 MB", level="INFO")

    # Check if dataset paths exist
    for path in args.path_data:
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist.")

    if args.exclude is not None:
        subjects_to_exclude = []
        with open(args.exclude, "r") as f:
            file = yaml.safe_load(f)
            subjects_to_exclude += file['chunks']
            subjects_to_exclude += file['stitched']

        excluded_subs = [re.search(r'(sub-m\d{6})', subject).group(0) for subject in subjects_to_exclude]
        excluded_subs = list(set(excluded_subs))
        logger.info(f"Excluded subjects: {excluded_subs}")
        logger.info(f"Number of excluded subjects: {len(excluded_subs)}")

    # Single site
    sites = list(LABEL_SUFFIXES.keys())
    for site in sites:
        create_directories(path_out, site)  # create 

    # get all the lesion files from the json file that PL created and shared
    path_json = args.path_json
    all_lesion_files_testing_large = []
    if path_json is not None:
        with open(path_json, "r") as f:
            file = json.load(f)
            for split in ["train", "validation", "test"]:
                for case in file[split]:
                    if case['site'] == 'sct-testing-large':
                        all_lesion_files_testing_large.append(case['label'])

    all_lesion_files, train_images, test_images =[], {}, {}
    subjects_tum, subjects_testing_large = [], []
    # temp dict for storing dataset commits
    dataset_commits = {}

    # loop over the datasets
    for dataset in args.path_data:
        root = Path(dataset)

        # get the git branch and commit ID of the dataset
        dataset_name = os.path.basename(os.path.normpath(dataset))
        branch, commit = get_git_branch_and_commit(dataset)
        dataset_commits[dataset_name] = f"git-{branch}-{commit}"
        site_name = 'tum' if "bavaria" in dataset_name else 'testing-large'

        if site_name == 'tum':
            # get recursively all GT '_label-lesion' files
            lesion_label_suffix = LABEL_SUFFIXES[site_name][1]
            lesion_files = [str(path) for path in root.rglob(f'*_{lesion_label_suffix}.nii.gz')]
            # for straigteneing preprocessing, chunk-4 images were removed because there was no SC to straighten
            # remove the chunk-4 images from the list
            subjects_tum = [fetch_subject_nifti_details(file)[0] for file in lesion_files]
            all_lesion_files.extend(lesion_files)
        else:
            subjects_testing_large = [fetch_subject_nifti_details(file)[0] for file in all_lesion_files_testing_large]
            all_lesion_files.extend(all_lesion_files_testing_large)

    # Get the training and test splits
    # NOTE: we need a patient-wise split (not image-wise split) to ensure that the same patient is not 
    # present in both training and test sets
    subjects_tum = sorted(list(set(subjects_tum)))
    subjects_testing_large = sorted(list(set(subjects_testing_large)))
    
    # exclude the subjects that are in the exclude list
    if args.exclude is not None:
        subs_tum = [sub for sub in subjects_tum if sub not in excluded_subs]

    logger.info(f"Found subjects in the tum dataset: {len(subs_tum)}")
    logger.info(f"Found subjects in the sct-testing-large dataset: {len(subjects_testing_large)}")        

    tr_subs_tum, te_subs_tum = train_test_split(subs_tum, test_size=test_ratio, random_state=args.seed)

    tr_subs_testing_large, te_subs_testing_large = train_test_split(
        subjects_testing_large, test_size=test_ratio, random_state=args.seed)
    
    for sub in tr_subs_tum+tr_subs_testing_large:
        # get the lesion files for the subject
        lesion_files_sub = [file for file in all_lesion_files if sub in file]

        for lesion_file in lesion_files_sub:
            if not os.path.exists(lesion_file):
                logger.info(f"Lesion file {lesion_file} does not exist. Skipping.")
                continue
            # add the lesion file to the training set
            train_images[lesion_file] = lesion_file

    logger.info(f"len test set TUM: {len(te_subs_tum)}")
    logger.info(f"len test set TESTING_LARGE: {len(te_subs_testing_large)}")
    for sub in te_subs_tum+te_subs_testing_large:
        # get the lesion files for the subject
        lesion_files_sub = [file for file in all_lesion_files if sub in file]

        for lesion_file in lesion_files_sub:
            if not os.path.exists(lesion_file):
                logger.info(f"Lesion file {lesion_file} does not exist. Skipping.")
                continue
            # add the lesion file to the test set
            test_images[lesion_file] = lesion_file

    logger.info(f"Total images in the training set (combining all datasets): {len(train_images)}")
    logger.info(f"Total images in the test set (combining all datasets): {len(test_images)}")
    logger.info(f"len(all lesion files): {len(all_lesion_files)}")
    # print version of each dataset in a separate line
    for dataset_name, dataset_commit in dataset_commits.items():
        logger.info(f"{dataset_name} dataset version: {dataset_commit}")

    # Counters for train and test sets
    train_ctr, test_ctr = 0, 0
    train_niftis, test_nifitis = [], []
    # Loop over all images
    for subject_label_file in tqdm(all_lesion_files, desc="Iterating over all images"):
        
        subject_id = fetch_subject_nifti_details(subject_label_file)[0]
        if subject_id.startswith("sub-m"):
            site_name = 'tum'
            # Construct path to the background image
            subject_image_file = subject_label_file.replace('/derivatives/labels', '').replace(f'_{LABEL_SUFFIXES[site_name][1]}', '')

        else:
            site_name = 'testing-large'
            # Construct path to the background image
            subject_image_file = subject_label_file.replace('/derivatives/labels', '').replace(f'_{LABEL_SUFFIXES[site_name][1]}', '')

            # also get the image and label from git-annes
            # NOTE: sct-testing-large has a lot of images which might/might not have labels. 
            # Get only those images which have labels and are present in the dataframe (and belong to the pathology)
            dataset_path = "/home/GRAMES.POLYMTL.CA/u114716/datasets/sct-testing-large"
            gitannex_cmd_label = f'cd {dataset_path}; git annex get {subject_label_file}'
            gitannex_cmd_image = f'cd {dataset_path}; git annex get {subject_image_file}'
            
            # download also the sc seg file
            subject_seg_file = subject_label_file.replace(f'_{LABEL_SUFFIXES[site_name][1]}', f'_{LABEL_SUFFIXES[site_name][0]}')
            gitannex_cmd_seg = f'cd {dataset_path}; git annex get {subject_seg_file}'
            try:
                subprocess.run(gitannex_cmd_image, shell=True, check=True)
                subprocess.run(gitannex_cmd_label, shell=True, check=True)
                subprocess.run(gitannex_cmd_seg, shell=True, check=True)
                logger.info(f"Downloaded {os.path.basename(subject_image_file)} from git-annex")
                logger.info(f"Downloaded {os.path.basename(subject_label_file)} from git-annex")
                logger.info(f"Downloaded {os.path.basename(subject_seg_file)} from git-annex")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error in downloading {file} from git-annex: {e}")

        # Train images
        if subject_label_file in train_images.keys():

            train_ctr += 1
            # add the subject image file to the list of training niftis
            train_niftis.append(os.path.basename(subject_image_file))

            # create the new convention names for nnunet
            sub_name = f"{str(Path(subject_image_file).name).replace('.nii.gz', '')}"

            subject_image_file_nnunet = os.path.join(path_out_imagesTr,
                                                        f"{args.dataset_name}_{sub_name}_{train_ctr:03d}_0000.nii.gz")
            subject_label_file_nnunet = os.path.join(path_out_labelsTr,
                                                    f"{args.dataset_name}_{sub_name}_{train_ctr:03d}.nii.gz")

            # use region-based labels if required
            if args.region_based:
                # overwritten the subject_label_file with the region-based label
                subject_label_file = get_region_based_label(subject_label_file, subject_image_file,
                                                            site_name, sub_name, thr=0.5)
                if subject_label_file is None:
                    print(f"Skipping since the region-based label could not be generated")
                    continue

            # copy the files to new structure
            shutil.copyfile(subject_image_file, subject_image_file_nnunet)
            shutil.copyfile(subject_label_file, subject_label_file_nnunet)

            # convert the image and label to RPI using the Image class
            image = Image(subject_image_file_nnunet)
            image.change_orientation("RPI")
            image.save(subject_image_file_nnunet)

            label = Image(subject_label_file_nnunet)
            label.change_orientation("RPI")
            label.save(subject_label_file_nnunet)

            # remove the temporary region-based label file
            if args.region_based:
                os.remove(subject_label_file)

            # don't binarize the label if either of the region-based or multi-channel training is set
            if not args.region_based:
                binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)

        # Test images
        elif subject_label_file in test_images:

            test_ctr += 1
            # add the image file to the list of testing niftis
            test_nifitis.append(os.path.basename(subject_image_file))

            # create the new convention names for nnunet
            sub_name = f"{str(Path(subject_image_file).name).replace('.nii.gz', '')}"

            subject_image_file_nnunet = os.path.join(Path(path_out, f'imagesTs_{site_name}'),
                                                     f'{args.dataset_name}_{sub_name}_{test_ctr:03d}_0000.nii.gz')
            subject_label_file_nnunet = os.path.join(Path(path_out, f'labelsTs_{site_name}'),
                                                     f'{args.dataset_name}_{sub_name}_{test_ctr:03d}.nii.gz')
            
            # use region-based labels if required
            if args.region_based:
                # overwritten the subject_label_file with the region-based label
                subject_label_file = get_region_based_label(subject_label_file, subject_image_file,
                                                            site_name, sub_name, thr=0.5)

                if subject_label_file is None:
                    continue

            # copy the files to new structure
            shutil.copyfile(subject_image_file, subject_image_file_nnunet)
            shutil.copyfile(subject_label_file, subject_label_file_nnunet)

            image = Image(subject_image_file_nnunet)
            image.change_orientation("RPI")
            image.save(subject_image_file_nnunet)

            label = Image(subject_label_file_nnunet)
            label.change_orientation("RPI")
            label.save(subject_label_file_nnunet)

            # print(f"\nCopying {subject_image_file} to {subject_image_file_nnunet}")
            # convert the image and label to RPI using the Image class
            
            # remove the temporary region-based label file
            if args.region_based:
                os.remove(subject_label_file)

            # don't binarize the label if either of the region-based or multi-channel training is set
            if not args.region_based:
                # to do native label?
                binarize_label(subject_image_file_nnunet, subject_label_file_nnunet)

        else:
            print("Skipping file, could not be located in the Train or Test splits split.", subject_label_file)

    logger.info(f"----- Dataset conversion finished! -----")
    logger.info(f"Number of training and validation images (across all sites): {train_ctr}")
    # Get number of train and val images per site
    train_images_per_site = {}
    for train_subject in train_images:
        site = sites[0]
        if site in train_images_per_site:
            train_images_per_site[site] += 1
        else:
            train_images_per_site[site] = 1
    # Print number of train images per site
    for site, num_images in train_images_per_site.items():
        logger.info(f"Number of training and validation images in {site}: {num_images}")

    logger.info(f"Number of test images (across all sites): {test_ctr}")
    # Get number of test images per site
    test_images_per_site = {}
    for test_subject in test_images:
        site = sites[0]
        if site in test_images_per_site:
            test_images_per_site[site] += 1
        else:
            test_images_per_site[site] = 1
    # Print number of test images per site
    for site, num_images in test_images_per_site.items():
        logger.info(f"Number of test images in {site}: {num_images}")

    # create the yaml file containing the train and test niftis
    create_yaml(train_niftis, test_nifitis, path_out, args, train_ctr, test_ctr, dataset_commits)


if __name__ == "__main__":
    main()