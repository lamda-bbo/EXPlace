# EXPlace: Expertise Can Be Helpful for Reinforcement Learning-based Macro Placement

This floder includes: **placement DEF files** and **full evaluation flows** that generates the performance results shown in Table 3. 

We would like to promote replication and convenient validation of our results. 

## Placement DEF files

The placement DEF files are given by Google Drive:
- [OpenROAD cases](https://drive.google.com/file/d/1gh-KZTQDRNnc7q_2KYXGmPMJQkkRMcjj/view?usp=sharing)

## Evaluation of OpenROAD

This evaluation flow is adopted from [OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts).

### Installation of OpenROAD

OpenROAD is necessary for OpenROAD cases' evaluation. We highly recommend the installation of [OpenROAD binary](https://github.com/Precision-Innovations/OpenROAD/releases/tag/2.0-17198-g8396d0866). 

Download a suitable version from the link above, and it can be then installed by:
```
sudo apt install ./openroad_2.0_amd64-ubuntu20.04.deb
```
The environment will be automatically resolved during the installation.

### Prepare DEFs

First navigate to the directory `ICLR26-EXPlace/OpenROAD-flow`. 

Then put the provided [OpenROAD DEF files](https://drive.google.com/file/d/1gh-KZTQDRNnc7q_2KYXGmPMJQkkRMcjj/view?usp=sharing) at `ICLR26-EXPlace/OpenROAD-flow/OpenROAD_DEFs.tar.gz`. Then unpack it by:
```
tar -xzvf OpenROAD_DEFs.tar.gz
```

### Run test

Run single case:
```
bash run_EXPlace.sh ariane133
bash run_EXPlace.sh ariane136
bash run_EXPlace.sh black_parrot
bash run_EXPlace.sh bp_be_top
bash run_EXPlace.sh bp_fe_top
bash run_EXPlace.sh swerv_wrapper
```
Run all six cases:
```
bash run.sh
```
Here `run.sh` contains six commands that can be run dependently for all six cases reported in our reported table. 

After successfully running the whole evaluation flow, the PPA metrics are shown in

```
ICLR26-EXPlace/OpenROAD-flow/logs/nangate45/{design_name}/EXPlace/6_final.json
```

