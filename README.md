# Hybrid Former
Implementation of [ A Convolutional Neural Network -Transformer Architecture for Low Dose Computed Tomography Image Denoising (Hybrid Former)] 
* The input image patch(64x64 size) is extracted randomly from the 512x512 size image. --> 
* use Adam optimizer
* NVIDIA GeForce RTX 3080 series GPU
* [DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15094096.svg)](https://doi.org/10.5281/zenodo.15094096)

----
### DATASET  (For downloding the DATASET, please visit the link bellow) 
The 2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge by Mayo Clinic   
[https://www.aapm.org/GrandChallenge/LowDoseCT/](https://ctcicblog.mayo.edu/2016-low-dose-ct-grand-challenge/) and to download the dataset please visit : https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h

The `data path` structure:


    data path
    ├── L067
    │   ├── quarter_3mm
    │   │       ├── L067_QD_3_1.CT.0004.0001 ~ .IMA
    │   │       ├── L067_QD_3_1.CT.0004.0002 ~ .IMA
    │   │       └── ...
    │   └── full_3mm
    │           ├── L067_FD_3_1.CT.0004.0001 ~ .IMA
    │           ├── L067_FD_3_1.CT.0004.0002 ~ .IMA
    │           └── ...
    ├── L096
    │   ├── quarter_3mm
    │   │       └── ...
    │   └── full_3mm
    │           └── ...      
    ...
    │
    └── L506
        ├── quarter_3mm
        │       └── ...
        └── full_3mm
                └── ...     

 
