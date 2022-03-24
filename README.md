# EvoKG: Jointly Modeling Event Time and Network Structure for Reasoning over Temporal Knowledge Graphs
This repository provides the code and data of the paper
["EvoKG: Jointly Modeling Event Time and Network Structure for Reasoning over Temporal Knowledge Graphs"](https://arxiv.org/abs/2202.07648), Namyong Park, Fuchen Liu, Purvanshi Mehta, Dana Cristofor, Christos Faloutsos, and Yuxiao Dong, The Fifteenth ACM International Conference on Web Search and Data Mining (WSDM) 2022.

## Setup
Run [`script/setup_evokg.sh`](./script/setup_evokg.sh) to create a conda environment named `evokg` and install required packages.

## Datasets
Datasets used in our paper can be found in the [`data`](./data/) folder. No additional data preprocessing is needed to run the code.

## Running EvoKG
Scripts in [`script/link_pred/`](./script/link_pred/) and [`script/time_pred/`](./script/time_pred/) can be used to run EvoKG for temporal link prediction and event time prediction, respectively.
Execution logs and results are stored in the `result` folder by default. To save results in a different folder, update settings.py accordingly.

## Citing
If you use the code or datasets in this repository, please cite our paper.
```bibtex
@inproceedings{park2022evokg,
    title={{EvoKG}: Jointly Modeling Event Time and Network Structure for Reasoning over Temporal Knowledge Graphs},
    author={Namyong Park and Fuchen Liu and Purvanshi Mehta and Dana Cristofor and Christos Faloutsos and Yuxiao Dong},
    booktitle={{WSDM}},
    year={2022},
}
```
