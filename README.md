# SNACS Course Project

## The Effect of Class Imbalance when Under-sampling in Supervised Link Prediction

This repository contains the code developed by António Bezerra and Inês Silva for the 2022 edition of Leiden University's SNACS course.

Datasets used are not included in the ZIP file, however, they can be freely downloaded from the KONECT project website. Our code assumes a certain directory structure for datasets. All datasets should be placed in a folder named `data` which then contains a folder named `datasets`. Inside that folder, extract each dataset into a subfolder with the following names:
- `mit` - [http://konect.cc/networks/mit/](http://konect.cc/networks/mit/)
- `munmun_twitterex_ut` - [http://konect.cc/networks/munmun_twitterex_ut/](http://konect.cc/networks/munmun_twitterex_ut/)
- `prosper-loans` - [http://konect.cc/networks/prosper-loans/](http://konect.cc/networks/prosper-loans/)
- `topology` - http://konect.cc/networks/topology/

Inside the data folder, also create a directory called `clean_datasets` so that our files can output to it.

Each file contained in this repository was created with the following goals:

- `extract_data.py` generates the data for supervised link prediction from a network dataset;
- `extract_data_condmat.py` generates the data for supervised link prediction from the COND-MAT dataset;
- `extract_data_bipartite.py` generates the data for supervised link prediction from a bipartite network dataset;
- `network_features.py` contains helper functions for feature extraction;
- `train.py` performs link prediction on the desired dataset using the specified undersampling technique;
- `graph_properties.ipynb` was used for gaining insights into the properties of the used datasets;
- `visualize_results.ipynb` was used for creating the plots and tables for the final paper.

Necessary python requirements can be found in the `requirements.txt` file.