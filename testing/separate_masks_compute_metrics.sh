#!/bin/bash

label_type=$1   # "lesion" or "sc"
# check if the label_type is provided
if [ -z $label_type ]; then
    echo "Please provide the label type: 'lesion' or 'sc'"
    exit 1
fi

if [ $label_type == "lesion" ]; then
    echo "Separating region-based masks into lesions only"
    threshold=1
    metrics_to_compute="dsc nsd rel_vol_error lesion_ppv lesion_sensitivity lesion_f1_score ref_count pred_count"
else
    echo "Separating region-based masks into spinal cord+lesions"
    threshold=0
    metrics_to_compute="dsc nsd rel_vol_error"
fi

dataset_name="Dataset901_tumMSChunksRegion"

# if dataset_name is Dataset901_tumMSChunksRegion, then the site is "muc_stacked"
if [ ${dataset_name} == "Dataset901_tumMSChunksRegion" ]; then
    site="muc_stacked"
    # site="muc"
elif [ ${dataset_name} == "Dataset902_tumMSStitchedRegion" ]; then
    site="muc"
elif [ ${dataset_name} == "Dataset903_tumMSChunksStraightRegion" ]; then
    site="muc_native_stacked"
elif [ ${dataset_name} == "Dataset904_tumMSStitchedStraightRegion" ]; then
    site="muc_native"
elif [ ${dataset_name} == "Dataset910_tumMSChunksPolyNYUAxialRegion" ]; then
    # site="tum_stacked"
    site="ucsf"
else
    echo "Unknown dataset name: ${dataset_name}"
    exit 1
fi

models=("2d" "3d_fullres")
chunk="chunk-3"


# Separating GT masks into cord and lesion masks

path_gt=${nnUNet_raw}/${dataset_name}/labelsTs_${site}
path_out=${nnUNet_raw}/${dataset_name}/labelsTs_${site}_${label_type}_${chunk}

if [ ! -d ${path_out} ]; then
    mkdir -p ${path_out}
fi

echo "====================================================================================================="
echo "Separating GROUND-TRUTH masks into spinal cord and lesion masks"
echo "====================================================================================================="

for file in ${path_gt}/*_${chunk}_T2w*.nii.gz; do

    file_out=${path_out}/$(basename ${file})
    file_out=${file_out/.nii.gz/_${label_type}.nii.gz}

    sct_maths -i ${file} -bin ${threshold} -o ${file_out}

done


# ======================================================================================================================
# Separating region-based predictions into separate masks
# ======================================================================================================================

folds=(0) # 1 2)
for model in ${models[@]}; do

    for fold in ${folds[@]}; do

        # echo "-----------------------------------------------------------------------------------------------------"
        # echo "Separating region-based masks for ${site} of ${dataset_name} with ${model} model and fold ${fold}"
        # echo "-----------------------------------------------------------------------------------------------------"

        path_predictions=${nnUNet_results}/${dataset_name}/nnUNetTrainerDiceCELoss_noSmooth__nnUNetPlans__${model}/fold_${fold}/test_${site}
        path_out_separated=${path_predictions}_${label_type}_${chunk}

        # if the output directory does not exist, create it
        if [ ! -d ${path_out_separated} ]; then
            mkdir -p ${path_out_separated}
        fi

        for file in ${path_predictions}/*_${chunk}_T2w*.nii.gz; do

            file_out=${path_out_separated}/$(basename ${file})
            file_out=${file_out/.nii.gz/_${label_type}.nii.gz}

            sct_maths -i ${file} -bin ${threshold} -o ${file_out}
        done

        # Compute MetricsReloaded for the separated masks
        echo "====================================================================================================="
        echo "Computing MetricsReloaded for ${site} of ${dataset_name} with ${model} model and fold ${fold}"
        echo "====================================================================================================="
        
        python ~/tum-poly/MetricsReloaded/compute_metrics_reloaded.py \
            -reference ${nnUNet_raw}/${dataset_name}/labelsTs_${site}_${label_type}_${chunk} \
            -prediction ${path_out_separated} \
            -output ${path_out_separated}/metrics_final_${label_type}_${chunk}.csv \
            -metrics ${metrics_to_compute} \
            -jobs 8
    done

done

