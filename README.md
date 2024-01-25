# PerfVec

PerfVec is a deep learning-based performance modeling framework that learns
high-dimensional, independent/orthogonal program and microarchitecture
representations.
Once learned, a program representation can be used to predict its performance
on any microarchitecture, and likewise, a microarchitecture representation can
be applied in the performance prediction of any program.
Additionally, PerfVec yields a foundation model that captures the performance
essence of instructions, which can be directly used by developers in numerous
performance modeling related tasks without incurring its training cost.

## Folder

DP: data processing scripts.

ML: machine learning scripts for training, testing, etc.

CFG: configurations of various datasets.

<!---
## Data Processing
```
source setup.sh
```

## Data Processing

### Combine data set.
```
python -m DP.combine_mmap -n <number of files>
```

### Calculate data set normalization factors.
```
python -m DP.norm
```

## Datasets

0: cache access levels
1: reuse distance
-->
