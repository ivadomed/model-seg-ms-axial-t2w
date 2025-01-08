"""
Script to run the stitching.py script for all subjects and sessions in the Bavaria dataset. 
Fetches all teh axial and sagittal T2w images, as well as the lesions-manual and seg-manual images, and runs the stitching.py script for each of them.

Author: Julian McGinnis
"""

import os
import subprocess
import argparse

def get_files(directory, keyword):
    """
    Get a list of .nii.gz files that contain a certain keyword in their name.
    """
    return [f for f in os.listdir(directory) if f.endswith('.nii.gz') and keyword in f]

def main(root_dir, script_dir):
    # Loop over all subjects
    for subject in os.listdir(root_dir):
        sub_dir = os.path.join(root_dir, subject)
        
        # Loop over all sessions for each subject
        for session in os.listdir(sub_dir):
            ses_dir = os.path.join(sub_dir, session, 'anat')
            
            # Get all axial T2w images
            t2w_files = get_files(ses_dir, 'acq-ax_chunk')

            # filter t2w files, so we do not have lesion or seg in them
            t2w_files = [x for x in t2w_files if "seg" not in x]
            t2w_files = [x for x in t2w_files if "les" not in x]
            
            # Run stitching script for T2w images
            if t2w_files:
                output_file = '{}_{}_acq-ax_T2w.nii.gz'.format(subject, session)
                output_file_path = os.path.join(ses_dir, output_file)
                command = ['python3', os.path.join(script_dir, 'stitching.py'), '-v', '-no_bias', '-no_hist', '--images'] + t2w_files + ['--output', output_file_path]
                print(command)
                subprocess.run(command, cwd=ses_dir)

            # Get all lesions-manual files
            lesions_files = get_files(ses_dir, 'lesions-manual_T2w.nii.gz')
            
            # Run stitching script for lesions-manual files
            if lesions_files:
                output_file = '{}_{}_lesions-manual_T2w.nii.gz'.format(subject, session)
                output_file_path = os.path.join(ses_dir, output_file)
                command = ['python3', os.path.join(script_dir, 'stitching.py'), '-v', '-no_bias', '-no_hist','-seg','--images'] + lesions_files + ['--output', output_file_path]
                print(command)
                subprocess.run(command, cwd=ses_dir)

            # Get all seg-manual files
            seg_files = get_files(ses_dir, 'seg-manual_T2w.nii.gz')
            
            # Run stitching script for seg-manual files
            if seg_files:
                output_file = '{}_{}_seg-manual_T2w.nii.gz'.format(subject, session)
                output_file_path = os.path.join(ses_dir, output_file)
                command = ['python3', os.path.join(script_dir, 'stitching.py'), '-no_bias', '-no_hist', '-v', '-seg','--images'] + seg_files + ['--output', output_file_path]
                print(command)
                subprocess.run(command, cwd=ses_dir)
            
            # Get all sagittal T2w images
            t2w_files = get_files(ses_dir, 'acq-sag_chunk')

            # filter t2w files, so we do not have lesion or seg in them
            t2w_files = [x for x in t2w_files if "seg" not in x]
            t2w_files = [x for x in t2w_files if "les" not in x]

            # Run stitching script for sagittal T2w images
            if t2w_files:
                output_file = '{}_{}_acq-sag_T2w.nii.gz'.format(subject, session)
                output_file_path = os.path.join(ses_dir, output_file)
                command = ['python3', os.path.join(script_dir, 'stitching.py'), '-v', '-no_bias', '-no_hist', '--images'] + t2w_files + ['--output', output_file_path]
                print(command)
                subprocess.run(command, cwd=ses_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and stitch Nifti images.")
    parser.add_argument('--root_dir', type=str, help='Root directory where the data is located.')
    parser.add_argument('--script_dir', type=str, help='Directory where the stitching.py script is located.')
    args = parser.parse_args()

    main(args.root_dir, args.script_dir)
