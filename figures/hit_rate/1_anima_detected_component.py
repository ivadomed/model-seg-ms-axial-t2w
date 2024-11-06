import os
import subprocess
import argparse

def run_anima_detected_components(gt_directory, pred_directory):
    # Collect all gt and pred files from their respective directories
    gt_files = sorted([f for f in os.listdir(gt_directory) if f.endswith('.nii.gz') if "chunk-4" not in f])
    pred_files = sorted([f for f in os.listdir(pred_directory) if f.endswith('.nii.gz') if "chunk-4" not in f])

    # Assert that the number of gt and pred files is the same
    assert len(gt_files) == len(pred_files), "The number of ground truth and prediction files must be the same!"
    
    for gt_file, pred_file in zip(gt_files, pred_files):
       
        assert os.path.basename(gt_file).replace('.nii.gz', '') == os.path.basename(pred_file).replace('.nii.gz', ''), "Ground truth and prediction files must have the same base name!"
        # Extract relevant information from the file names to create the output csv file name
        base_name = gt_file.replace('.nii.gz', '')
        output_csv = f'detected_components_{base_name}.csv'

        # Run the animaDetectedComponents command
        gt_path = os.path.join(gt_directory, gt_file)
        pred_path = os.path.join(pred_directory, pred_file)
        output_csv_path = os.path.join(pred_directory, output_csv)  # Save the CSV in the pred directory

        command = [
            'animaDetectedComponents', 
            '-t', pred_path, 
            '-r', gt_path, 
            '-o', output_csv_path
        ]

        print(f'Running: {" ".join(command)}')
        subprocess.run(command)

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Run animaDetectedComponents with separate directories for ground truth and predicted NIfTI files.")
    parser.add_argument('--gt', type=str, required=True, help="Directory containing the ground truth NIfTI files")
    parser.add_argument('--pred', type=str, required=True, help="Directory containing the predicted NIfTI files")

    args = parser.parse_args()

    # Run the script with the provided directories
    run_anima_detected_components(args.gt, args.pred)
