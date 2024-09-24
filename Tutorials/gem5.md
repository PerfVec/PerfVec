## Install and run gem5 for PerfVec

1. Download gem5 and switch to `riscv` branch.

```
git clone https://github.com/lingda-li/gem5.git
cd gem5
git checkout riscv
```

2. Build gem5.
gem5 depends scons and some other libraries.
Check gem5 documents if there are problems during installation.

```
scons -j4 build/RISCV/gem5.opt
```

3. Use gem5 to simulate a program run.

```
./build/RISCV/gem5.opt configs/example/riscv/starter_se.py --cpu=minor <command line to execute the simulated program>
```

Use a simple matrix multiplication program as an example:

```
./build/RISCV/gem5.opt configs/example/riscv/starter_se.py --cpu=minor "<path to PerfVec>/Tutorials/mm 8 8 8"
```
