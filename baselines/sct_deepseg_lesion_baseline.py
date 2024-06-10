import os
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Function to find all .nii.gz files in a folder
def find_niftis(folder):
    return sorted(glob.glob(os.path.join(folder, '**', '*.nii.gz'), recursive=True))

# Function to create output directory if it doesn't exist
def create_output_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# Function to process a single image-label pair
def process_image_label_pair(image_path, label_path, output_folder):

    # assert image and label name are identical if "_0000" is stripped from image name
    assert os.path.basename(image_path).replace('_0000', '') == os.path.basename(label_path), "Image and label names must be identical."

    # Run sct_deepseg_lesion
    subprocess.run(['sct_deepseg_lesion', '-i', image_path, '-c', 't2_ax', '-ofolder', output_folder, '-centerline', 'svm', '-brain', '0'])

    filename = os.path.basename(image_path)
    output_segmentation_path = os.path.join(output_folder, filename.replace('.nii.gz', '_lesionseg.nii.gz'))

    # Define output csv file path
    output_csv = os.path.join(output_folder, filename.replace('.nii.gz', '.csv'))

    # Run MetricsReloaded compute_metrics_reloaded.py
    #subprocess.run([
    #    'python3', os.path.expanduser(f'~/git_repositories/MetricsReloaded/compute_metrics_reloaded.py'),
    #    '-reference', label_path,
    #    '-prediction', output_segmentation_path,
    #    '-output', output_csv
    #])

    # Skipping metrics computation for now
    print("Done.")

# Main function
def main(images_folder, labels_folder, output_folder, num_workers):

    image_list = find_niftis(images_folder)
    label_list = find_niftis(labels_folder)

    # Assert that the number of images and labels is the same
    assert len(image_list) == len(label_list), "The number of image and label files must be identical."

    # Create output folder
    create_output_folder(output_folder)

    # Parallel processing of image-label pairs
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        tasks = []
        for image_path, label_path in zip(image_list, label_list):
            tasks.append(executor.submit(process_image_label_pair, image_path, label_path, output_folder))

        # Use tqdm to display a progress bar
        for task in tqdm(tasks):
            task.result()

if __name__ == "__main__":
    import argparse

    # Argument parsing
    parser = argparse.ArgumentParser(description="Process nifti files and compute metrics.")
    parser.add_argument('--images', required=True, help='Path to the images folder.')
    parser.add_argument('--labels', required=True, help='Path to the labels folder.')
    parser.add_argument('--output', required=True, help='Path to the output prediction folder.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers.')

    args = parser.parse_args()

    # Run the main function
    main(args.images, args.labels, args.output, args.num_workers)
