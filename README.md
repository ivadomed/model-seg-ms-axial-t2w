# Assessing the Impact of Spinal Cord Curvature in Axial T2-weighted Intramedullary MS Lesion Segmentation

This repository contains the code for deep learning-based segmentation of the spinal cord and intramedullary MS lesions in Axial T2-weighted MRI scans. The model is based on the [nnUNetv2 framework](https://github.com/MIC-DKFZ/nnUNet). This project is a collaboration between NeuroPoly (Polytechnique Montreal, Quebec) and TUM (Munich, Bavaria)

## Model Overview

The model was trained on raw T2-weighted axial images of MS patients from multiple (four) sites. The TUM dataset is longitudinal (two sessions) and consisted of individual chunks (cervical, thoracic and lumbar) covering the entire spine. The three other sites used in this study were taken from the private `sct-testing-large` dataset from NeuroPoly. To ensure uniformity across sites, all images were initially re-oriented to RPI. Given an input image, the model is able to segment *both* the lesion and the spinal cord.

TODO: add a figure here

## Using the model

### Install dependencies

- [Spinal Cord Toolbox (SCT) v6.2](https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases/tag/6.2) or higher -- follow the installation instructions [here](https://github.com/spinalcordtoolbox/spinalcordtoolbox?tab=readme-ov-file#installation)
- [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) 
- Python (v3.9)

Once the dependencies are installed, download the latest model:

```bash
sct_deepseg -install-task seg_sc_lesion_t2w_ms
```

### Getting the lesion and spinal cord segmentation

To segment a single image, run the following command: 

```bash
sct_deepseg -i <INPUT> -task seg_sc_lesion_t2w_ms
```

For example:

```bash
sct_deepseg -i sub-001_T2w.nii.gz -task seg_sc_lesion_t2w_ms
```

The outputs will be saved in the same directory as the input image, with the suffix `_lesion_seg.nii.gz` for the lesion 
and `_sc_seg.nii.gz` for the spinal cord.

## Analysis Pipeline

TODO: 

## Citation Info

If you find this work and/or code useful for your research, please cite our paper:

TODO: