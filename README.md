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

## Use Cases

### <a name="learnrep"></a> Learning a program representation using a pretrained foundation model

A concrete step-by-step example can be found in
[Tutorials/learn_rep.md](Tutorials/learn_rep.md).

1. <a name="gem5"></a> Get the instruction execution trace using gem5.
The modified gem5 can be obtained from
[https://github.com/lingda-li/gem5/tree/ml_sim](https://github.com/lingda-li/gem5/tree/ml_sim).
Simulating a program in SE mode using this gem5 will generate two instruction
trace files, `trace.txt` and `sq.trace.txt`.

2. <a name="inputgen"></a> Generate the PerfVec model input from the gem5 instruction execution trace.

    `python -m DP.trace2nmmap <trace name>`

3. Create a config file for the generated data.
An example can be seen in [CFG/com_spectest_1223.py](CFG/com_spectest_1223.py).
Make a copy of it, and then put the input path and size in `sim_datasets`.

4. Run the trained PerfVec model.
The following command will generate a representation file
`res/prep_<cfg_name>_<checkpoint_name>.pt` in the format of PyTorch tensor,
which can be used for performance prediction.

    ```
    python -m ML.test --sbatch --no-save --sim-length=<# instructions> --cfg=<config name> \\
      --rep --checkpoints=<pretrained model checkpoint> <pretrained model instantiation>
    ```

### Making performance prediction

With both program and microarchitecture representations, PerfVec can predict
the performance when a program runs on a microarchitecture.
A concrete step-by-step example can be found in
[Tutorials/predict.md](Tutorials/predict.md).

1. Create a config file for the generated data.
An example can be seen in [CFG/rep_spectest_0124.py](CFG/rep_spectest_0124.py).
Put the program representation file in `dataset`.

2. Make execution time prediction with a pre-trained model that includes
microarchitecture representations using the following command.

    ```
    python -m ML.test --no-save --cfg=<config name> --pred \\
      --checkpoints=<microarchitecture representation checkpoint> "Predictor(cfg,bias=True)"
    ```

    Alternatively, you can use the following command to make performance
    prediction directly from an instruction trace instead of program
    representations.

    ```
    python -m ML.test --sbatch --no-save --sim-length=<# instructions> --cfg=<config name> \\
      --sim --checkpoints=<pretrained model checkpoint> <pretrained model instantiation>
    ```

### Training a foundation model

This is for more advanced users.

1. Pick programs for training.
Any programs can be used theorectically, and we use SPEC CPU 2017 benchmarks in the paper.

2. Get their instruction execution traces on many sampled architectures using gem5.
This is similar to [Step 1](#gem5) of the first example.

3. Generate a dataset for PerfVec.
For each program that is intended to be used in training or testing, do the following.
    * Generate an input file. The process is identical to [Step 2](#inputgen) of the first example.
    * Generate an output file that combines instruction latencies on all
    microarchitectures which serves as prediction targets, and convert it to
    the numpy memmap format used in training, using the following commands.

        ```
        ./DP/buildComOut <trace on uarch 1> <trace on uarch 2> ... <trace on uarch n>
        python -m DP.inst2mmap -t -l=<# instructions in combined output> <combined output>
        ```


5. Create a config file for the generated data.
An example can be seen in [CFG/com_0522.py](CFG/com_0522.py).
The following information is needed in the config file.
    * List the name, input size, and output size of all programs in `datasets`.
    * Modify `data_set_idx` to be the number of programs.
    * Calculate the total number of entries (i.e., sum output sizes), and then modify
    `testbatchnum`, `testbatchsize`, `validbatchnum`, and `validbatchsize` to
    specify the testing and validation portions.
    * Modify `cfg_num` to be the number of microarchitectures.

6. Train a PerfVec foundation model.
See [ML/models](ML/models) for various model options, or implement your own.

    ```
    python -m ML.train --cfg=<config name> --train-size=<# instructions for training> \\
      --epochs=<# epochs> --batch-size=<batch size> --sbatch <model instantiation>
    ```

## Pretained Foundation Models

| Model                | Instantiation                       | Link                                                              |
|----------------------|-------------------------------------|-------------------------------------------------------------------|
| LSTM-2-256 (default) | InsLSTM(256,2,narchs=77,bias=False) | https://github.com/PerfVec/PerfVecDB/blob/main/LSTM_256_2_1222.pt |

## Requirement

Python 3, PyTorch (>= 1.10 recommended), numpy, ptflops

## Code Structure

DP: C++ and Python data processing scripts.
To build these scripts, `cd DP; make in; make out`

ML: machine learning scripts for training, testing, etc.

CFG: configurations of various datasets.

DA: data analysis scripts.

<!---
-->

## Contact

Please leave your questions in GitHub issues or direct them to [Lingda Li](mailto:lli@bnl.gov).

