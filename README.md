# PerfVec

PerfVec is a deep learning-based performance modeling framework that learns
high-dimensional, independent/orthogonal program and microarchitecture
representations.
Once learned, a program representation can be used to predict its performance
on any microarchitecture, and likewise, a microarchitecture representation can
be applied in the performance prediction of any program.
Additionally, PerfVec yields foundation models that captures the performance
essence of instructions, which can be directly used by developers in numerous
performance modeling related tasks without incurring the training cost.
More details can be found in our paper at
[https://arxiv.org/abs/2310.16792](https://arxiv.org/abs/2310.16792).

## Example

### Learning the representation of a program using a pretrained foundation model

1. Get the instruction execution trace using gem5.
The modified gem5 can be obtained from
[https://github.com/lingda-li/gem5/tree/ml_sim](https://github.com/lingda-li/gem5/tree/ml_sim).
Simulating a program in SE mode using this gem5 will generate two instruction
trace files, `trace.txt` and `trace.sq.txt`.

2. Generate the PerfVec model input from the gem5 instruction execution trace.

`python -m DP.trace2nmmap <input trace>`

3. Create a config file for the generated data.

An example can be seen in `CFG/com_spectest_1223.py`.
Make a copy of it, and then put the input path and size in `sim_datasets`.

4. Run the trained PerfVec model.

```
python -m ML.test --sbatch --no-save --sim-length=<# instructions> --cfg=<config file in CFG> \\
  --rep --checkpoints=<pretrained model checkpoint> <pretrained model instantiation>
```

It will generate a representation file `res/prep_<cfg_name>_<checkpoint_name>.pt` in the format of PyTorch tensor, which can be used for performance prediction.

Alternatively, you can use the following command to directly make performance prediction with a pre-trained model that includes microarchitecture representations.

```
python -m ML.test --sbatch --no-save --sim-length=<# instructions> --cfg=<config file in CFG> \\
  --sim --checkpoints=<pretrained model checkpoint> <pretrained model instantiation>
```

### Training a foundation model

1. Pick programs for training.

2. Get their instruction execution traces on many sampled architectures using gem5.

3. Generate a training dataset from all gem5 instruction execution traces.

4. Create a config file for the generated data.

An example can be seen in `CFG/com_0922.py`.
Make a copy of it, and then put the path and size of training data in `datasets`.
Modify `data_set_idx` to be the length of used datasets.
Calculate the total number of instructions, and then modify `testbatchnum`, `testbatchsize`, `validbatchnum`, and `validbatchsize` to specify the testing and validation portions.

5. Train a PerfVec foundation model.

```
python -m ML.train --cfg=<config file in CFG> --epochs=<# epochs> --train-size=<# instructions for training> \\
  --batch-size=<batch size> --sbatch <model instantiation>
```

## Pretained Foundation Models

| Model                | Instantiation                       | Link                                                              |
|----------------------|-------------------------------------|-------------------------------------------------------------------|
| LSTM-2-256 (default) | InsLSTM(256,2,narchs=77,bias=False) | https://github.com/PerfVec/PerfVecDB/blob/main/LSTM_256_2_1222.pt |

## Folder

DP: data processing scripts.

ML: machine learning scripts for training, testing, etc.

CFG: configurations of various datasets.

DA: data analysis scripts.

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

## Contact

Please leave your questions in GitHub issues or direct them to [Lingda Li](lli@bnl.gov).

