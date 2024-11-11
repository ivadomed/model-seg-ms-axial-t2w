#!/bin/bash
# Training nnUNet on just the Zurich dataset


path_data_dir=$1

# get the base name of the path_data_dir
dataset_name=$(basename $path_data_dir)

if [ $dataset_name == "Dataset901_tumMSChunksRegion" ]; then
    # site="muc_stacked_lesion"
    site="muc_lesion_chunk-3"
    training_type="single-channel"
    # NOTE: we have already separated lesion and sc masks for MetricsReloaded, 
    # so using the same niftis. `single-channel` arg directly computes anima metrics
    # without separating the masks

elif [ $dataset_name == "DeepSegLesionInference_tumNeuropoly" ]; then
    site="tum_deepseg-lesion_stacked"
    training_type="single-channel"

elif [ $dataset_name == "Dataset902_tumMSStitchedRegion" ]; then
    site="muc"
    training_type="region-based"

elif [ $dataset_name == "Dataset903_tumMSChunksStraightRegion" ]; then
    site="muc_native_stacked"
    training_type="region-based"

elif [ $dataset_name == "Dataset904_tumMSStitchedStraightRegion" ]; then
    site="muc_native"
    training_type="region-based"

elif [ $dataset_name == "Dataset905_tumMSChunksMulti" ]; then
    site="muc_stacked"
    training_type="multi-channel"

elif [ $dataset_name == "Dataset906_tumMSStitchedMulti" ]; then
    site="muc"
    training_type="multi-channel"

elif [ $dataset_name == "Dataset907_tumMSChunksStraightMulti" ]; then
    site="muc_native_stacked"
    training_type="multi-channel"

elif [ $dataset_name == "Dataset908_tumMSStitchedStraightMulti" ]; then
    site="muc_native"
    training_type="multi-channel"

else
    echo "Unknown dataset name: ${dataset_name}"
    exit 1
fi

nnunet_trainer=("nnUNetTrainerDiceCELoss_noSmooth")

# configurations=("3d_fullres" "2d")
configurations=("2d") # "3d_fullres")

# folds=(2 1 0)
folds=(0) # 1 2)

for configuration in ${configurations[@]}; do

    for fold in ${folds[@]}; do

        echo "-----------------------------------------------------------------------------------------------------"
        echo "Running ANIMA evaluation on ${site} of ${dataset_name} with ${configuration} model and fold ${fold}"
        echo "-----------------------------------------------------------------------------------------------------"
    
        python ~/tum-poly/miccai_amai_lesions_spine/testing/compute_anima_metrics.py \
            --pred-folder ~/nnunet-v2/nnUNet_results/${dataset_name}/${nnunet_trainer}__nnUNetPlans__${configuration}/fold_${fold}/test_${site} \
            --gt-folder ${nnUNet_raw}/${dataset_name}/labelsTs_${site} \
            --training-type ${training_type}
        # python ~/tum-poly/miccai_amai_lesions_spine/testing/compute_anima_metrics.py \
        #     --pred-folder ~/nnunet-v2/nnUNet_results/${dataset_name}/test_${site} \
        #     --gt-folder ${nnUNet_raw}/Dataset910_tumMSChunksPolyNYUAxialRegion/labelsTs_tum_for_deepseg_stacked \
        #     --training-type ${training_type}
    
    done

done
