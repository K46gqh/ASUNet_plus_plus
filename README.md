## ASU-Net++:
A nested U-Net with adaptive featureextractions for liver tumor segmentation

## Paper

This repository is the Keras implementation of ASU-Net++ in the paper below:

**ASU-Net++: A nested U-Net with adaptive feature extractions for liver tumor segmentation** <br/>
Qinhan Gao, Mohamed Almekkawy<br/>
Penn State University <br/>
Computers in Biology and Medicine <br/>
[paper](https://www.sciencedirect.com/science/article/abs/pii/S0010482521004820) | [code](https://github.com/K46gqh/ASUNet_plus_plus)

## Usage

- root
    * subCT\
        * 1
        * 2
        * ..
    * val\
    * data.py
    * model.py
    
Run the data.py to generate .npy file using the data samples in subCT and val directories for training, validation and testing. 
The model can be trained by directly running model.py afterward

## Citation
If you use UNet++ for your research, please cite the paper:

@article{GAO2021104688,
title = {ASU-Net++: A nested U-Net with adaptive feature extractions for liver tumor segmentation},
journal = {Computers in Biology and Medicine},
volume = {136},
pages = {104688},
year = {2021},
issn = {0010-4825},
doi = {https://doi.org/10.1016/j.compbiomed.2021.104688},
url = {https://www.sciencedirect.com/science/article/pii/S0010482521004820},
author = {Qinhan Gao and Mohamed Almekkawy},
keywords = {Ultrasound segmentation, CT segmentation, Tumor segmentation, Deep learning, Convolutional neural network}