#!/bin/bash
#
# Compare the CSA of soft GT thresholded at different values on spine-generic test dataset.
# 
# Adapted from: https://github.com/ivadomed/model_seg_sci/blob/main/baselines/comparison_with_other_methods_sc.sh
#
# Usage:
#     sct_run_batch -config config.json
#
# Example of config.json:
# {
#  "path_data"   : "<PATH_TO_DATASET>",
#  "path_output" : "<PATH_TO_DATASET>/results_qc_other_datasets/qc-reports",
#  "script"      : "<PATH_TO_REPO>/qc_other_datasets/generate_qc.sh",
#  "jobs"        : 5,
#  "script_args" : "<DATASET_TO_QC> <PATH_TO_REPO>/nnUnet/run_inference_single_subject.py <PATH_TO_NNUNET_MODEL> <PATH_TO_REPO>/monai/run_inference_single_image.py <PATH_TO_MONAI_MODEL> <PATH_TO_SWINUNETR_MODEL> <PATH_TO_MEDNEXT_MODEL>"
# }
#
# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"
#
# Author: Jan Valosek and Naga Karthik
#

# # Uncomment for full verbose
# set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Print retrieved variables from the sct_run_batch script to the log (to allow easier debug)
echo "Retrieved variables from from the caller sct_run_batch:"
echo "PATH_DATA: ${PATH_DATA}"
echo "PATH_DATA_PROCESSED: ${PATH_DATA_PROCESSED}"
echo "PATH_RESULTS: ${PATH_RESULTS}"
echo "PATH_LOG: ${PATH_LOG}"
echo "PATH_QC: ${PATH_QC}"

SUBJECT=$1
QC_DATASET=$2           # dataset name to generate QC for
PATH_NNUNET_SCRIPT=$3   # path to the nnUNet contrast-agnostic run_inference_single_subject.py
PATH_NNUNET_MODEL_1=$4    # path to the nnUNet chunks single site model
PATH_NNUNET_MODEL_2=$5    # path to the nnUNet chunks two sites model
# PATH_MONAI_SCRIPT=$5    # path to the MONAI contrast-agnostic run_inference_single_subject.py

echo "SUBJECT: ${SUBJECT}"
echo "QC_DATASET: ${QC_DATASET}"
echo "PATH_NNUNET_SCRIPT: ${PATH_NNUNET_SCRIPT}"
echo "PATH_NNUNET_MODEL_1: ${PATH_NNUNET_MODEL_1}"
echo "PATH_NNUNET_MODEL_2: ${PATH_NNUNET_MODEL_2}"

# ------------------------------------------------------------------------------
# CONVENIENCE FUNCTIONS
# ------------------------------------------------------------------------------

# Get ANIMA binaries path
anima_binaries_path=$(grep "^anima = " ~/.anima/config.txt | sed "s/.* = //" | sed 's/\/$//')

# Compute ANIMA segmentation performance metrics
compute_anima_metrics(){
  local FILESEG="$1"
  local FILEGT="$2"

  # We have to copy qform matrix from seg-manual to the automatically generated segmentation to avoid ITK error:
  # "Description: ITK ERROR: SegmentationMeasuresImageFilter(): Inputs do not occupy the same physical space!"
  # Related to the following issue : https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4135
  sct_image -i ${FILEGT}.nii.gz -copy-header ${FILESEG}.nii.gz -o ${FILESEG}_updated_header.nii.gz

  # Compute ANIMA segmentation performance metrics
  # -i : input segmentation
  # -r : GT segmentation
  # -o : output file
  # -d : surface distances evaluation
  # -s : compute metrics to evaluate a segmentation
  # -X : stores results into a xml file.
  ${anima_binaries_path}/animaSegPerfAnalyzer -i ${FILESEG}_updated_header.nii.gz -r ${FILEGT}.nii.gz -o ${PATH_RESULTS}/${FILESEG} -d -s -X

  rm ${FILESEG}_updated_header.nii.gz
}

# Copy GT segmentation (located under derivatives/labels)
copy_gt_seg(){
  local file="$1"
  # local label_suffix="$2"
  # Construct file name to GT segmentation located under derivatives/labels
  FILESEG="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file}_seg-manual.nii.gz"
  FILELESION="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file}_lesion-manual.nii.gz"
  echo ""
  echo "Looking for manual segmentation: $FILESEG"
  if [[ -e $FILESEG ]]; then
      echo "Found! Copying ..."
      rsync -avzh $FILESEG ${file}_seg-manual.nii.gz
      rsync -avzh $FILELESION ${file}_lesion-manual.nii.gz
      # rsync -avzh ${FILESEG/.nii.gz/.json} ${file}_seg-manual.json
  else
      echo "File ${FILESEG}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: Manual Segmentation ${FILESEG} does not exist. Exiting."
      exit 1
  fi

  # Create lesion QC
  sct_qc -i ${file}.nii.gz -s ${file}_seg-manual.nii.gz -d ${file}_lesion-manual.nii.gz -p sct_deepseg_lesion -plane axial -qc ${PATH_QC} -qc-subject ${SUBJECT}

}

# Segment spinal cord using methods available in SCT (sct_deepseg_sc or sct_propseg), resample the prediction back to
# native resolution and compute CSA in native space
segment_sc() {
  local file="$1"
  local method="$2"     # deepseg or propseg
  local contrast="$3"   # used for input arg `-c`
  local kernel="2d"     # 2d or 3d; only relevant for deepseg

  # Segment spinal cord
  if [[ $method == 'deepseg' ]];then
      FILESEG="${file}_seg_${method}_${kernel}"

      # Get the start time
      start_time=$(date +%s)
      # Run SC segmentation
      sct_deepseg_sc -i ${file}.nii.gz -o ${FILESEG}.nii.gz -c ${contrast} -kernel ${kernel} -qc ${PATH_QC} -qc-subject ${SUBJECT}
      # Get the end time
      end_time=$(date +%s)
      # Calculate the time difference
      execution_time=$(python3 -c "print($end_time - $start_time)")
      echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv
  fi

}

segment_lesion() {
  local file="$1"
  local method="$2"     # deepseg or propseg
  local contrast="$3"   # used for input arg `-c`; should be t2_ax
  # Segment spinal cord
  if [[ $method == 'deepseg_lesion' ]];then
      FILESEG="${file}_seg_${method}"

      # Get the start time
      start_time=$(date +%s)
      # Run Lesion segmentation
      sct_deepseg_lesion -i ${file}.nii.gz -o ${FILESEG}.nii.gz -c ${contrast} -centerline svm
      # Get the end time
      end_time=$(date +%s)
      # Calculate the time difference
      execution_time=$(python3 -c "print($end_time - $start_time)")
      echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv
      # # qc for lesion segmentation
      # sct_qc -i ${file}.nii.gz -s ${file}_seg-manual.nii.gz -d ${FILESEG}.nii.gz -p sct_deepseg_lesion -plane axial -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi

}


# Segment spinal cord using the contrast-agnostic nnUNet model
segment_nnUNet(){
  local file="$1"
  local model="$2"     # chunks_single or chunks_two
  local kernel="$3"     # 2d or 3d
  local label_type="$4"       # sc or lesion

  FILESEG="${file}_seg_${model}_${label_type}_${kernel}"
  echo $FILESEG

  if [[ ${model} == 'chunks_single' ]]; then
    PATH_NNUNET_MODEL=${PATH_NNUNET_MODEL_1}
  elif [[ ${model} == 'chunks_two' ]]; then
    PATH_NNUNET_MODEL=${PATH_NNUNET_MODEL_2}
  fi

  # Get the start time
  start_time=$(date +%s)
  # Run lesion segmentation
  python ${PATH_NNUNET_SCRIPT} -i ${file}.nii.gz -o ${FILESEG}.nii.gz -path-model ${PATH_NNUNET_MODEL}/nnUNetTrainerDiceCELoss_noSmooth__nnUNetPlans__${kernel} -pred-type ${label_type} -use-gpu
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # also get the spinal cord segmentation
  FILESC="${file}_seg_${model}_sc_${kernel}"
  python ${PATH_NNUNET_SCRIPT} -i ${file}.nii.gz -o ${FILESC}.nii.gz -path-model ${PATH_NNUNET_MODEL}/nnUNetTrainerDiceCELoss_noSmooth__nnUNetPlans__${kernel} -pred-type sc -use-gpu

  # Generate QC report
  sct_qc -i ${file}.nii.gz -s ${FILESC}.nii.gz -d ${FILESEG}.nii.gz -p sct_deepseg_lesion -plane axial -qc ${PATH_QC} -qc-subject ${SUBJECT}

  # # Compute CSA from the prediction resampled back to native resolution using the GT vertebral labels
  # sct_process_segmentation -i ${FILESEG}.nii.gz -vert 2:4 -vertfile ${file_gt_vert_label}_labeled.nii.gz -o $PATH_RESULTS/csa_label_types_c24.csv -append 1

}

# Segment spinal cord using the MONAI contrast-agnostic model
segment_sc_MONAI(){
  local file="$1"
  # local label_type="$2"     # soft or soft_bin
  local model="$2"     # monai, swinunetr, mednext

	# if [[ $label_type == 'soft' ]]; then
	# 	FILEPRED="${file}_seg_monai_soft_input"
	# 	PATH_MODEL=${PATH_MONAI_MODEL_SOFT}
	
	# elif [[ $label_type == 'bin' ]]; then
  #   FILEPRED="${file}_seg_monai_bin_input"
	# 	PATH_MODEL=${PATH_MONAI_MODEL_BIN}
	
	# fi
	if [[ $model == 'v2x' ]]; then
		FILEPRED="${file}_seg_${model}"
		PATH_MODEL=${PATH_MONAI_MODEL_1}
    max_feat=320
  
  elif [[ $model == 'v2x_contour' ]]; then
    FILEPRED="${file}_seg_${model}"
    PATH_MODEL=${PATH_MONAI_MODEL_2}
    max_feat=384
  
  elif [[ $model == 'v2x_contour_dcm' ]]; then
    FILEPRED="${file}_seg_${model}"
    PATH_MODEL=${PATH_MONAI_MODEL_3}
    max_feat=384
	
	# elif [[ $model == 'swinunetr' ]]; then
  #   FILEPRED="${file}_seg_swinunetr"
  #   PATH_MODEL=${PATH_SWIN_MODEL}
  	
	fi

  # Get the start time
  start_time=$(date +%s)
  # Run SC segmentation
  # python ${PATH_MONAI_SCRIPT} --path-img ${file}.nii.gz --path-out . --chkp-path ${PATH_MODEL} --device gpu --model ${model}
  python ${PATH_MONAI_SCRIPT} --path-img ${file}.nii.gz --path-out . --chkp-path ${PATH_MODEL} --device gpu --model monai --pred-type soft --pad-mode edge --max-feat ${max_feat}
  # Rename MONAI output
  mv ${file}_pred.nii.gz ${FILEPRED}.nii.gz
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILEPRED},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # # Generate QC report on soft predictions
  # sct_qc -i ${file}.nii.gz -s ${FILEPRED}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

  # Binarize MONAI output (which is soft by default); output is overwritten
  sct_maths -i ${FILEPRED}.nii.gz -bin 0.5 -o ${FILEPRED}.nii.gz

  # Generate QC report 
  sct_qc -i ${file}.nii.gz -s ${FILEPRED}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}

  # compute ANIMA metrics
  compute_anima_metrics ${FILEPRED} ${file}_seg-manual

}


# ------------------------------------------------------------------------------
# SCRIPT STARTS HERE
# ------------------------------------------------------------------------------
# get starting time:
start=`date +%s`

# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Note: we use '/./' in order to include the sub-folder 'ses-0X'
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
if [[ $QC_DATASET == "bavaria-quebec-spine-ms-unstitched" ]]; then

  label_suffix="seg-manual"
  # Copy source T2w images
  # Note: we use '/./' in order to include the sub-folder 'ses-0X'
  # We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
  rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}_*ax_chunk-*.nii.gz .
  # rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}_*ax_T2w.nii.gz .

  # Go to subject folder for source images
  cd ${SUBJECT}/anat

  # ------------------------------------------------------------------------------
  # T2w
  # ------------------------------------------------------------------------------
  # run a counter from 0 to 4 for the chunks
  # for i in {1..3}; do
  for i in 1; do
      # Define T2w file
      file_t2="${SUBJECT//[\/]/_}_acq-ax_chunk-${i}_T2w"
      # file_t2="${SUBJECT//[\/]/_}_acq-ax_T2w"

      # Copy GT spinal cord segmentation
      copy_gt_seg "${file_t2}" "${label_suffix}"

      # Check if file_t2 exists
      if [[ ! -e ${file_t2}.nii.gz ]]; then
          echo "File ${file_t2}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
          echo "ERROR: File ${file_t2}.nii.gz does not exist. Exiting."
          exit 1
      fi

  done

elif [[ $QC_DATASET == "sct-testing-large" ]]; then

  label_suffix="lesion-manual"
  contrast="acq-ax_T2w"

  # Copy source images
  rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}*${contrast}.* .

  # Go to the folder where the data is
  cd ${PATH_DATA_PROCESSED}/${SUBJECT}/anat

  # Get file name
  file_t2="${SUBJECT}_${contrast}"

  # Copy GT spinal cord segmentation
  copy_gt_seg "${file_t2}" "${label_suffix}"

  # Check if file exists
  if [[ ! -e ${file_t2}.nii.gz ]]; then
      echo "File ${file_t2}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: File ${file_t2}.nii.gz does not exist. Exiting."
      exit 1
  fi

else
  echo "ERROR: Dataset ${QC_DATASET} not recognized. Exiting."
  exit 1

fi

# Segment SC using different methods and compute ANIMA segmentation performance metrics
CUDA_VISIBLE_DEVICES=2 segment_nnUNet "${file_t2}" "chunks_single" '2d' 'lesion'
CUDA_VISIBLE_DEVICES=3 segment_nnUNet "${file_t2}" "chunks_single" '3d_fullres' 'lesion'

CUDA_VISIBLE_DEVICES=0 segment_nnUNet "${file_t2}" "chunks_two" '2d' 'lesion'
CUDA_VISIBLE_DEVICES=1 segment_nnUNet "${file_t2}" "chunks_two" '3d_fullres' 'lesion'

segment_lesion "${file_t2}" 'deepseg_lesion' 't2_ax'
# segment_sc "${file_t2}" 't2' 'deepseg' '2d'
# segment_sc "${file_t2}" 't2' 'deepseg' '3d'



# ------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------

# Display results (to easily compare integrity across SCT versions)
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
