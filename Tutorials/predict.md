### Making performance prediction

Finish Steps 1-5 in [the tutorial of representation learning](learn_rep.md),
and then use the following command to make performance prediction directly from
an instruction trace instead of program representations.
Before executing the command, remove `#` before `from .custom_data_in import *`
and comment out `from .custom_data_inout import *` in
[ML/custom_data.py](../ML/custom_data.py).

```
python -m ML.test --no-test --sbatch --no-save --sim-length=1000000000 --cfg=mm_cfg \\
  --sim --select --checkpoints=checkpoints/LSTM_256_2_1222.pt "InsLSTM(256,2,narchs=77,bias=False)"
```

This command will output the predicted execution time in the unit of 0.1 ns,
which can be compared with the gem5 simulation results (`simSeconds` in
`stats.txt`) to validate PerfVec results.

<!---
Please complete [the tutorial of representation learning](learn_rep.md) before
proceeding.

1. Create a config file for the generated data.
An example can be seen in `CFG/rep_spectest_0124.py`.
Put the program representation file in `dataset`.

2. Make execution time prediction with a pre-trained model that includes
microarchitecture representations using the following command.

```
python -m ML.test --no-save --cfg=<config name> --pred \\
  --checkpoints=<microarchitecture representation checkpoint> "Predictor(cfg,bias=True)"
```
-->
