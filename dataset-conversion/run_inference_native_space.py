import os
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pathlib import Path
import nibabel as nib
import numpy as np

# Function to find all .nii.gz files in a folder
def find_niftis(folder):
    return sorted(glob.glob(os.path.join(folder, '**', '*.nii.gz'), recursive=True))

# Function to create output directory if it doesn't exist
def create_output_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# Function to process a single prediction
def process_prediction(pred_file, image_file, warp_field, output_dir):
    base_name = os.path.basename(pred_file).replace('.nii.gz', '')
    pred_sc = os.path.join(os.path.dirname(pred_file), base_name + '_cord.nii.gz')
    pred_lesion = os.path.join(os.path.dirname(pred_file), base_name + '_lesion.nii.gz')

    # Binarize the prediction file for cord and lesion
    subprocess.run(['sct_maths', '-i', pred_file, '-bin', '0', '-o', pred_sc])
    subprocess.run(['sct_maths', '-i', pred_file, '-bin', '1', '-o', pred_lesion])

    # Apply warping field to the cord and lesion files
    warped_lesion = os.path.join(output_dir, base_name + '_lesion_warped.nii.gz')
    warped_cord = os.path.join(output_dir, base_name + '_cord_warped.nii.gz')
    
    subprocess.run(['sct_apply_transfo', '-i', pred_lesion, '-d', image_file, '-w', warp_field, '-o', warped_lesion])
    subprocess.run(['sct_apply_transfo', '-i', pred_sc, '-d', image_file, '-w', warp_field, '-o', warped_cord])

    # Threshold the warped files
    thresholded_lesion = warped_lesion.replace('.nii.gz', '_thr.nii.gz')
    thresholded_cord = warped_cord.replace('.nii.gz', '_thr.nii.gz')
    
    subprocess.run(['sct_maths', '-i', warped_lesion, '-bin', '0.5', '-o', thresholded_lesion])
    subprocess.run(['sct_maths', '-i', warped_cord, '-bin', '0.5', '-o', thresholded_cord])

    # Combine the thresholded files into a final segmentation using nibabel
    cord_img = nib.load(thresholded_cord)
    lesion_img = nib.load(thresholded_lesion)

    cord_data = cord_img.get_fdata()
    lesion_data = lesion_img.get_fdata()

    combined_data = np.zeros_like(cord_data)
    combined_data[cord_data > 0] = 1
    combined_data[lesion_data > 0] = 2

    combined_segmentation = os.path.join(output_dir, base_name + '_combined.nii.gz')
    combined_img = nib.Nifti1Image(combined_data, cord_img.affine, cord_img.header)
    nib.save(combined_img, combined_segmentation)

def main(prediction_dir, images_dir, warping_fields_dir, output_dir, num_workers):
    prediction_files = find_niftis(prediction_dir)
    image_files = find_niftis(images_dir)
    warping_field_files = find_niftis(warping_fields_dir)

    # Create output folder
    create_output_folder(output_dir)

    assert len(prediction_files) == len(image_files), 'Invalid number of predictions'

    # Parallel processing of image-label pairs
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        tasks = []
        for prediction_path, image_path, warping_path  in zip(prediction_files, image_files, warping_field_files):
            tasks.append(executor.submit(process_prediction, prediction_path, image_path, warping_path, output_dir))

        # Use tqdm to display a progress bar
        for task in tqdm(tasks):
            task.result()

if __name__ == "__main__":
    import argparse

    # Argument parsing
    parser = argparse.ArgumentParser(description="Process nifti files and compute metrics.")
    parser.add_argument('--prediction', required=True, help='Path to the prediction folder.')
    parser.add_argument('--images', required=True, help='Path to the images folder.')
    parser.add_argument('--warping_fields', required=True, help='Path to the warping fields folder.')
    parser.add_argument('--output', required=True, help='Path to the output folder.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers.')

    args = parser.parse_args()

    # Run the main function
    main(args.prediction, args.images, args.warping_fields, args.output, args.num_workers)
