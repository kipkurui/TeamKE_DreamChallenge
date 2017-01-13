# TeamKE submission to DREAM challenge


### Step 1: Install the requirements
The main dependencies are:
- numpy
- scipy
- hdf5
- h5py
- pybedtools
- pysam
- pybigwig
- pandas
- scikit-learn
- xgboost
- seaborn

We are using the conda environment in our analysis, 
which can be quickly set up with requirements.yml.

Additionally, you will need kcmeans clustering from FISIM. 
You can get it as part of [MARSTools](https://github.com/kipkurui/MARSTools) or from [FISIM](http://genome.ugr.es/fisim).


### Step 2: Get all the required data in place

This code uses the Unzipped files provided by the organisers. 
These folders should be set up and contain the individual datasets.
- annotations
- ChIPseq
- code
- DNASE
- Results 
 In addition to the provided data, the following additional data are used.

1. Motifs: Contains all motifs used in our code
    - .meme: Downloaded from various sources as indicated by name suffix
    - .tomtom: Their ranks based on similarity using TOMTOM
    - \_cluster.txt: Clustered suing FISim
2. DNAShape information downloaded from ftp://rohslab.usc.edu/hg19/
    - hg19.HelT.wig.bw
    - hg19.MGW.wig.bw
    - hg19.ProT.wig.bw
    - hg19.Roll.wig.bw
    
## Step 3: Run the code

The code can be run for the Ladder or final board as follows:

```
- python Dream_code_train.py TF-name stage: F, for final or L for ladderboard
- eg: python Dream_code_train.py ATF2 L

```
    
Run without variables for details. 
```
python Dream_code_train.py
```

