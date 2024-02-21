## Learning a program representation using a pretrained foundation model

1. See [gem5.md](gem5.md) to generate an example instruction trace.

2. Setup the environment and build data processing programs.

```
cd <path to PerfVec>
source setup.sh
cd DP; make in; cd ..
```

3. Generate the PerfVec model input from the gem5 instruction execution trace.

```
mv <path to gem5>/sq.trace.txt <path to gem5>/trace.sq.txt
python -m DP.trace2nmmap <path to gem5>/trace.txt
```

This command produces a file named `<path to gem5>/trace.in.nmmap` and outputs
the number of instructions it includes.

4. Create a config file for the generated data.
Modify [mm_cfg.py](mm_cfg.py) to fill in the number of instructions in the
trace and gem5 path.
Then copy it to the CFG folder: `cp Tutorials/mm_cfg.py CFG`.

5. Download the pretrained PerfVec model from
[https://github.com/PerfVec/PerfVecDB/blob/main/LSTM_256_2_1222.pt](https://github.com/PerfVec/PerfVecDB/blob/main/LSTM_256_2_1222.pt).

```
wget https://github.com/PerfVec/PerfVecDB/raw/main/LSTM_256_2_1222.pt -P checkpoints
```

6. Run the pretrained PerfVec model to learn and combine instruction
representations.
Before executing the following command, remove `#` before `from .custom_data_in
import *` and comment out `from .custom_data_inout import *` in
[ML/custom_data.py](../ML/custom_data.py).

```
python -m ML.test --no-test --sbatch --no-save --sim-length=1000000000 --cfg=mm_cfg \\
  --rep --checkpoints=checkpoints/LSTM_256_2_1222.pt "InsLSTM(256,2,narchs=77,bias=False)"
```

This command will generate a representation file
`res/prep_mm_cfg_LSTM_256_2_1222.pt` in the format of PyTorch tensor, which can
be used for performance prediction and analysis.
