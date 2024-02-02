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
More details can be found in our paper at
[https://arxiv.org/abs/2310.16792](https://arxiv.org/abs/2310.16792).

## Folder

DP: data processing scripts.

ML: machine learning scripts for training, testing, etc.

CFG: configurations of various datasets.

DA: data analysis scripts.

## Example

### Learn the representation of a program using a trained model

1. Get the instruction execution trace using gem5.
The modified gem5 can be obtained from
[https://github.com/lingda-li/gem5/tree/ml_sim](https://github.com/lingda-li/gem5/tree/ml_sim).
Simulating a program in SE mode using this gem5 will generate two instruction
trace files, `trace.txt` and `trace.sq.txt`.

2. Generate the PerfVec model input from the gem5 instruction execution trace.

`python -m DP.trace2nmmap <input trace>`

3. Create a config file for the generated data.
An example can be seen in `CFG/com_spectest_1223.py`.
Put the input path and size in `sim_datasets`.

4. Run the trained PerfVec model.

`python -m ML.test --sbatch --no-save --sim-length=<# instructions> --cfg=<config file in CFG>
  --rep --checkpoints=<pretrained model checkpoint> <pretrained model instantiation>`

## Pretained Foundation Models

| Model                | Instantiation                            | Link                                                              |
|----------------------|------------------------------------------|-------------------------------------------------------------------|
| LSTM-2-256 (default) | "InsLSTMRep(256,2,narchs=77,bias=False)" | https://github.com/PerfVec/PerfVecDB/blob/main/LSTM_256_2_1222.pt |

<!---
`./dp/buildQ a.txt a.sq.txt`
-->

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
