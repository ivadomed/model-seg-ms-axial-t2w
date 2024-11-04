"""
Loop across JSON sidecar files and nii headers in the input path and parse from them the following information:
    MagneticFieldStrength
    Manufacturer
    ManufacturerModelName
    ProtocolName
    PixDim
    SliceThickness

If JSON sidecar is not available (sci-paris), fetch only PixDim and SliceThickness from nii header.

Example usage:
    python utils/fetch_sequence_parameters.py -i /path/to/dataset -contrast T2w

Author: Jan Valosek
"""

import os
import re
import json
import argparse

import pandas as pd
import nibabel as nib
from loguru import logger

LIST_OF_PARAMETERS = [
    'MagneticFieldStrength',
    'Manufacturer',
    'ManufacturerModelName',
    'ProtocolName',
    'EchoTime',
    'RepetitionTime',
    ]


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='Loop across JSON sidecar files in the input path and parse from them relevant information.',
        prog=os.path.basename(__file__).strip('.py')
    )
    parser.add_argument(
        '-i',
        required=True,
        type=str,
        help='Path to dataset json file (dataset_T2w_ax_neuropoly_tum.json) containing the image/label keys.',
    )

    return parser


def parse_json_file(file_path):
    """
    Read the JSON file and parse from it relevant information.
    :param file_path:
    :return:
    """

    file_path = file_path.replace('.nii.gz', '.json')

    # Read the JSON file, return dict with n/a if the file is empty
    try:
        with open(file_path) as f:
            data = json.load(f)
    except:
        print(f'WARNING: {file_path} is empty.')
        return {param: "n/a" for param in LIST_OF_PARAMETERS}

    # Initialize an empty dictionary to store the parsed information
    parsed_info = {}

    if 'zurich' in file_path:
        # For sci-zurich, JSON file contains a list of dictionaries, each dictionary contains a list of dictionaries
        data = data['acqpar'][0]
    elif 'colorado' in file_path:
        data = data

    # Loop across the parameters
    for param in LIST_OF_PARAMETERS:
        try:
            parsed_info[param] = data[param]
        except:
            parsed_info[param] = "n/a"

    return parsed_info


def parse_nii_file(file_path):
    """
    Read nii file header using nibabel and to get PixDim and SliceThickness.
    We are doing this because 'PixelSpacing' and 'SliceThickness' can be missing from the JSON file.
    :param file_path:
    :return:
    """

    # Read the nii file, return dict with n/a if the file is empty
    try:
        img = nib.load(file_path)
        header = img.header
    except:
        print(f'WARNING: {file_path} is empty. Did you run git-annex get .?')
        return {param: "n/a" for param in ['PixDim', 'SliceThickness']}

    # Initialize an empty dictionary to store the parsed information
    parsed_info = {
        'PixDim': list(header['pixdim'][1:3]),
        'SliceThickness': float(header['pixdim'][3])
    }

    return parsed_info


def find_site_in_path(path):
    match = re.search(r'sub-m|sub-nyu|sub-bwh|sub-ucsf', path)
    if match:
        return 'TUM' if 'sub-m' in match.group() else 'NYU' if 'sub-nyu' in match.group() else 'BWH' if 'sub-bwh' in match.group() else 'UCSF'
    else:
        return 'Unknown'


def main():
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()
    path_out = "/home/GRAMES.POLYMTL.CA/u114716/tum-poly/sequence_parameters"
    if not os.path.exists(path_out):
        os.makedirs(path_out, exist_ok=True)
    logger.add(os.path.join(path_out, f"sequence_params.log"), rotation="10 MB", level="INFO")
    
    path_json = args.i

    list_of_files_per_site = {}    
    with open(path_json, "r") as f:
        file = json.load(f)
        for split in ["train", "validation", "test"]:
            for case in file[split]:
                site = find_site_in_path(case['image'])
                if site not in list_of_files_per_site:
                    list_of_files_per_site[site] = []
                list_of_files_per_site[site].append(case['image'])

    for site, list_of_files in list_of_files_per_site.items():

        logger.info(f"Site: {site}")
        # Loop across JSON sidecar files in the input path
        parsed_data = []
        for file in list_of_files:
            if file.endswith('.nii.gz'):
                # print(f'Parsing {file}')
                parsed_json = parse_json_file(file)
                parsed_header = parse_nii_file(file)
                # Note: **metrics is used to unpack the key-value pairs from the metrics dictionary
                parsed_data.append({'filename': file, **parsed_json, **parsed_header})

        # Create a pandas DataFrame from the parsed data
        df = pd.DataFrame(parsed_data)

        # Save the DataFrame to a CSV file
        df.to_csv(os.path.join(path_out, f'{site}_parsed_data.csv'), index=False)
        logger.info(f"Parsed data saved to {os.path.join(path_out, f'{site}_parsed_data.csv')}")

        if site == "TUM":
            # Print the min and max values of the MagneticFieldStrength, PixDim, and SliceThickness
            logger.info(df[['MagneticFieldStrength', 'PixDim', 'SliceThickness']].agg(['min', 'max']))
            logger.info(df[['PixDim', 'SliceThickness']].agg(['median']))

            # # Print unique values of the Manufacturer and ManufacturerModelName
            # print(df[['Manufacturer', 'ManufacturerModelName']].drop_duplicates())
            # Print number of filenames for unique values of the Manufacturer
            logger.info(df.groupby('Manufacturer')['filename'].nunique())
            # Print number of filenames for unique values of the MagneticFieldStrength
            logger.info(df.groupby('MagneticFieldStrength')['filename'].nunique())
            # print the EchoTime and RepetitionTime
            logger.info(df[['EchoTime', 'RepetitionTime']].agg(['median']))
            # # groupby echo time and repetition time
            # logger.info(df.groupby(['EchoTime', 'RepetitionTime'])['filename'].nunique())

        else:
            # NOTE: for the 3 sites from sct-testing-large, MagneticFieldStrength is missing, instead the 
            # strength is mentioned with the manufacturer name
            # Print the min and max values of the MagneticFieldStrength, PixDim, and SliceThickness
            logger.info(df[['PixDim', 'SliceThickness']].agg(['min', 'max']))
            # compute the median of the PixDim and SliceThickness
            

            # # Print unique values of the Manufacturer and ManufacturerModelName
            # print(df[['Manufacturer', 'ManufacturerModelName']].drop_duplicates())
            # Print number of filenames for unique values of the Manufacturer
            logger.info(df.groupby('Manufacturer')['filename'].nunique())
            # # Print number of filenames for unique values of the MagneticFieldStrength
            # logger.info(df.groupby('MagneticFieldStrength')['filename'].nunique())
            # drop rows were echo time or repetition time is missing
            df = df[df['EchoTime'] != 'n/a']
            # print the EchoTime and RepetitionTime
            logger.info(df[['EchoTime', 'RepetitionTime']].agg(['min', 'max']))
        


if __name__ == '__main__':
    main()
