#!/bin/bash
# Script to compute metrics reloaded across spinal cord segmentation models

# Run MetricsReloaded 
metrics_to_compute="dsc nsd rel_vol_error"
# methods_to_compare="monai_v2.0 nnunet_2d propseg deepseg_2d deepseg_3d"
methods_to_compare="monai_v2.5 monai_v2.6 propseg deepseg_2d"
PATH_RESULTS="/home/GRAMES.POLYMTL.CA/u114716/tum-poly/results_monai_vs_deepseg_20241031/results"

for method in ${methods_to_compare}; do

  for i in {1..3}; do

    echo "-----------------------------------------------------------------------------------------------------"
    echo "Running MetricsReloaded on ${method} method and chunk ${i}"
    echo "-----------------------------------------------------------------------------------------------------"

    python ~/tum-poly/MetricsReloaded/compute_metrics_reloaded.py \
        -reference ${PATH_RESULTS}/GTs/chunk-${i} \
        -prediction ${PATH_RESULTS}/preds/${method}/chunk-${i} \
        -output ${PATH_RESULTS}/preds/${method}/chunk-${i}/metrics_final_${method}_chunk-${i}_sc.csv \
        -metrics ${metrics_to_compute} \
        -jobs 8
  done

done
