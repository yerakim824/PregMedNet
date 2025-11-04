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


## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
Please contact Yeasul Kim (ykim824@stanford.edu) with any questions.