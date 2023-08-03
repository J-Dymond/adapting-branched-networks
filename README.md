# Class Separation

Repository for class-separation, CKA, and linear probe experiments in the paper (Adapting branched networks to realise
progressive intelligence)[https://bmvc2022.mpi-inf.mpg.de/0990.pdf]. 

The repository has the code and analysis files used in the paper, as well as some additional scripts which weren't used in the paper.

The code has not been updated since November 2022, however if there are any queries using the code don't desitate to contact (jd5u19@soton.ac.uk)[jd5u19@soton.ac.uk] for assistance. 

The key files are as such:

- code/branched-gradient-tracked-training.py
    - Trains chosen model on selected dataset for the desired number of repeats (runs) and epochs
- code/refactored-class-seperation.py
    - Records the class separation values on a model in the selected directory, on the selected dataset, for the desired number of runs in that directory
- code/linear-probes.py
    - Records the linear probe scores on a model in the selected directory, on the selected dataset, for the desired number of runs in that directory
- code/cka-analysis.py
    - Performs the CKA analysis between two models in the selected directories, on the selected dataset, for the desired number of runs

In the analysis folder there are a number of analysis python notebookes which use the outputs of above analysis files to produce the graphs used in the paper, there are also some additional experiments that didn't make it into the paper.