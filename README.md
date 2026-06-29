
# VHR-GSIS: Very High-Resolution Informal Settlement Mapping Dataset

This repository provides the code and project-specific scripts used for producing the **VHR-GSIS** dataset: a very-high-resolution informal-settlement mapping product for 37 Global South cities in 2010 and 2025.

The dataset and code are associated with the manuscript:

**A Very High-Resolution Dataset of Informal Settlements in 37 Global South Cities (2010–2025)**

## Overview

VHR-GSIS provides vector-based informal-settlement maps derived from Google Maps Level-19 satellite imagery, with an approximate spatial resolution of 0.59 m. The dataset contains informal-settlement distributions for two time points, 2010 and 2025, and includes a status attribute to facilitate temporal analysis.

The mapping workflow includes:

1. preparation of VHR image tiles and binary labels;
2. training city-specific semantic segmentation models;
3. large-area inference for target cities;
4. mosaicking and clipping model outputs to city boundaries;
5. vectorization and attribution of temporal status categories;
6. accuracy assessment and validation.

## Notice
The segmentation model used in this study is based on the open-source **MMSegmentation** framework. We do not redistribute the complete MMSegmentation source code in this repository. Instead, we provide the project-specific configuration files and model-related scripts required to reproduce our workflow. Users should install MMSegmentation separately from its official repository and place the provided files according to the directory structure described below. 

## Data

The released VHR-GSIS dataset is available through Zenodo: https://doi.org/10.5281/zenodo.18606153

## Status Attribute

The released vector files include a `status` field:

| Status value | Meaning |
|---|---|
| `0` | Stable informal settlement, present in both 2010 and 2025 |
| `1` | Disappeared informal settlement, present in 2010 but not in 2025 |
| `2` | Newly emerged informal settlement, present in 2025 but not in 2010 |

Users can reconstruct single-year informal-settlement maps as follows:

- **2010 informal-settlement extent**: select polygons with `status = 0` or `status = 1`;
- **2025 informal-settlement extent**: select polygons with `status = 0` or `status = 2`.

The status attribute is intended to facilitate data usage and temporal analysis. It is derived from the direct comparison of two independently generated single-epoch maps rather than from a separate change-detection model.

