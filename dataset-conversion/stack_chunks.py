import os
import argparse
import numpy as np
import nibabel as nib
import re
from tqdm import tqdm
from utils import Image

def load_nifti_image(file_path):
    """
    Construct absolute path to the nifti image, check if it exists, and load the image data.
    :param file_path: path to the nifti image
    :return: nifti image data
    """
    file_path = os.path.expanduser(file_path)   # resolve '~' in the path
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File {file_path} does not exist.')
    nifti_image = nib.load(file_path)
    return nifti_image.get_fdata()


def find_subject_session_chunk_in_path(path):
    """
    Extracts subject and session identifiers from the given path.
    :param path: Input path containing subject and session identifiers.
    :return: Extracted subject and session identifiers or None if not found.
    """
    # pattern = r'.*_(sub-m\d{6})_(ses-\d{8}).*_(chunk-\d{1})_.*'
    pattern = r'.*_(sub-m\d{6}_ses-\d{8}).*_(chunk-\d{1})_.*'
    match = re.search(pattern, path)
    if match:
        return match.group(1), match.group(2)
    else:
        return None, None, None


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Compute MetricsReloaded metrics for segmentation tasks.')

    # Arguments for model, data, and training
    parser.add_argument('-prediction', required=True, type=str,
                        help='Path to the folder with nifti images of test predictions or path to a single nifti image '
                             'of test prediction chunks.')
    parser.add_argument('-reference', required=True, type=str,
                        help='Path to the folder with nifti images of reference (ground truth) or path to a single '
                             'nifti image of reference chunks (ground truth).')

    return parser


def get_images_in_folder(prediction, reference):
    """
    Get all files (predictions and references/ground truths) in the input directories
    :param prediction: path to the directory with prediction files
    :param reference: path to the directory with reference (ground truth) files
    :return: list of prediction files, list of reference/ground truth files
    """
    # Get all files in the directories
    prediction_files = [os.path.join(prediction, f) for f in os.listdir(prediction) if f.endswith('.nii.gz')]
    reference_files = [os.path.join(reference, f) for f in os.listdir(reference) if f.endswith('.nii.gz')]
    # Check if the number of files in the directories is the same
    if len(prediction_files) != len(reference_files):
        raise ValueError(f'The number of files in the directories is different. '
                         f'Prediction files: {len(prediction_files)}, Reference files: {len(reference_files)}')
    print(f'Found {len(prediction_files)} files in the directories.')
    # Sort the files
    # NOTE: Hopefully, the files are named in the same order in both directories
    prediction_files.sort()
    reference_files.sort()

    return prediction_files, reference_files



def main():
    # parse command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Args.prediction and args.reference are paths to folders with multiple nii.gz files (i.e., MULTIPLE subjects)
    if os.path.isdir(args.prediction) and os.path.isdir(args.reference):
        # Get all files in the directories
        prediction_files, reference_files = get_images_in_folder(args.prediction, args.reference)
        prediction_files = sorted(prediction_files)
        reference_files = sorted(reference_files)

        path_out_pred = os.path.join(os.path.dirname(args.prediction), f'{os.path.basename(args.prediction)}_stacked')
        if not os.path.exists(path_out_pred):
            os.makedirs(path_out_pred, exist_ok=True)
        path_out_ref = os.path.join(os.path.dirname(args.reference), f'{os.path.basename(args.reference)}_stacked')
        if not os.path.exists(path_out_ref):
            os.makedirs(path_out_ref, exist_ok=True)

        # get the subject, session, and chunk identifiers from the path
        subjects_sessions = [find_subject_session_chunk_in_path(f)[0] for f in prediction_files if find_subject_session_chunk_in_path(f)]
        subjects_sessions = list(set(subjects_sessions))

        dataset_name = os.path.basename(prediction_files[0]).split('_')[0]

        for sub_ses in tqdm(subjects_sessions, desc='Computing metrics for each subject'):
            preds_per_sub_ses = [f for f in prediction_files if sub_ses in f]
            refs_per_sub_ses = [f for f in reference_files if sub_ses in f]

            preds_stack, refs_stack = [], []
            for pred, ref in zip(preds_per_sub_ses, refs_per_sub_ses):
                # load nifti images
                prediction_data = load_nifti_image(pred)
                reference_data = load_nifti_image(ref)

                # check whether the images have the same shape and orientation
                if prediction_data.shape != reference_data.shape:
                    raise ValueError(f'The prediction and reference (ground truth) images must have the same shape. '
                                        f'The prediction image has shape {prediction_data.shape} and the ground truth image has '
                                        f'shape {reference_data.shape}.')

                preds_stack.append(prediction_data)
                refs_stack.append(reference_data)

            # min_shape = np.min([pred.shape for pred in preds_stack], axis=0)
            max_shape = np.max([pred.shape for pred in preds_stack], axis=0)
            max_shape_ref = np.max([ref.shape for ref in refs_stack], axis=0)

            assert max_shape[0] == max_shape_ref[0], "The images must have the same shape at dim[0]"
            assert max_shape[1] == max_shape_ref[1], "The images must have the same shape at dim[1]"
            assert max_shape[2] == max_shape_ref[2], "The images must have the same shape at dim[2]"

            # pad the images to the same shape
            preds_stack = [np.pad(pred, ((0, max_shape[0] - pred.shape[0]), (0, max_shape[1] - pred.shape[1]), (0, max_shape[2] - pred.shape[2]))) for pred in preds_stack]
            refs_stack = [np.pad(ref, ((0, max_shape[0] - ref.shape[0]), (0, max_shape[1] - ref.shape[1]), (0, max_shape[2] - ref.shape[2]))) for ref in refs_stack]

            # stack the images
            preds_stacked = np.stack(preds_stack, axis=-1).astype(np.uint8)
            refs_stacked = np.stack(refs_stack, axis=-1).astype(np.uint8)

            # create a new file name for reference and prediction
            pred_fname = os.path.join(path_out_pred, f'{dataset_name}_{sub_ses}_preds_stack.nii.gz')
            ref_fname = os.path.join(path_out_ref, f'{dataset_name}_{sub_ses}_refs_stack.nii.gz')

            # save the stacked images as uint8
            nib.save(nib.Nifti1Image(preds_stacked, np.eye(4)), pred_fname)
            nib.save(nib.Nifti1Image(refs_stacked, np.eye(4)), ref_fname)


if __name__ == '__main__':
    main()
