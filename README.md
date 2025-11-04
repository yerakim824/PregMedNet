# PregMedNet
An implementation of **PregMedNet** from our publication: **Kim et al.** *"PregMedNet: Multifaceted Maternal Medication Impacts on Neonatal Complications"*

MedRxiv doi: https://doi.org/10.1101/2025.02.13.25322242

## Overview
<p align="center">
  <img src="figures/pregmednet_overview.png" width="80%">
</p>

**PregMedNet is a platform that provides multifaceted insights into the impacts of maternal medications during pregnancy on neonatal complications.**
We systematically analyzed large-scale medical claims data with machine-learning methods to estimate multifaceted perinatal medication impacts.

It includes:s
1. Estimation of maternal medication effects on neonatal outcomes, including both raw and counfounder-adjusted
2. Analysis of drug–drug interactions during pregnancy
3. Mechanism-of-action (MoA) inference through biological network integration

🔗 Comprehensive results are available on our interactive website: http://pregmednet.stanford.edu 

## Data Availability
This project utilizes the Merative™ MarketScan® Commercial Database, a real-world healthcare dataset that contains de-identified medical records from over 188 million patients across the United States.

The dataset is available for purchase by federal, nonprofit, academic, pharmaceutical, and other qualified research organizations.
Access to the data requires a data use agreement and purchase of the relevant subset needed for the study.

For more information on licensing the Merative™ MarketScan® Commercial Database, please visit:
👉 https://www.merative.com/documents/brief/marketscan-explainer-general

In this study, the database was accessed through the Stanford Center for Population Health Sciences (PHS). 

Detailed information about data processing and cohort construction can be found in the Methods section of our paper.

## Requirements
The code is written in Python3. Please install the packages present in the requiremtns.txt file. You may use:
```
pip install -r requirements.txt
```

## Code Structure
PregMedNet/
│
├── .devcontainer/                      # Development container configuration for reproducible environments
│
├── 1_medication_impact_calculation/    # Scripts for estimating medication–outcome associations
│   ├── benjamini_hochberg_correction.py      # Multiple testing correction (FDR control)
│   ├── drug_drug_interactions.py             # Analysis of concomitant drug–drug effects
│   ├── single_medication_impact_raw_odds.py  # Computes unadjusted (raw) odds ratios
│   └── single_medication_impact_adjusted_odds.py # Computes confounder-adjusted odds ratios
│
├── 2_mechanism-of-actions/             # Graph-based mechanism-of-action (MoA) inference
│   ├── MOA_functions.py                # Core functions for network integration and MoA computation
│   ├── MOA_only_with_protein_nodes.ipynb  # MoA inference using protein-level networks
│   └── MOA_with_biological_nodes.ipynb    # MoA inference integrating protein and biological nodes
│
├── Interactive_Webapp/                 # Source code for the interactive PregMedNet platform
│   ├── 2024_reference_tables/          # Reference data for node mapping (diseases, drugs, etc. kg.parquet file is also used files in 2_mechanism-of-actions folder)
│   ├── Dockerfile                      # Docker build for deployment
│   ├── cloudbuild.yaml                 # Google Cloud Build configuration for automated deployment
│   ├── PregMedNet_Functions.py         # Shared backend utility functions
│   ├── PregMedNet_Interactive_Website.py # Streamlit-based web interface (deployed at pregmednet.stanford.edu)
│   ├── requirements.txt                # Dependencies specific to the web app
│   └── README.md                       # Documentation for the web interface
│
├── figures/                            # Project figures for README and manuscript
│   └── pregmednet_overview.png
│
├── .gitattributes                      # Git LFS configuration for large files
├── LICENSE                             # License information
├── README.md                           # Project documentation
└── requirements.txt                    # Python package dependencies for 1_medication_impact_calculation and 2_mechanism-of-actions folders


## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
Please contact Yeasul Kim (ykim824@stanford.edu) with any questions.