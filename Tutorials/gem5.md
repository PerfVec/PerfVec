## Install and run gem5 for PerfVec

1. Download gem5 and switch to `ml_sim` branch.

```
git clone https://github.com/lingda-li/gem5.git
cd gem5
git checkout ml_sim
```

2. Build gem5.
gem5 depends scons and some other libraries.
Check gem5 documents if there are problems during installation.

```
scons -j4 build/ARM/gem5.opt
```

3. Use gem5 to simulate a program run.

```
./build/ARM/gem5.opt configs/example/arm/starter_se.py --cpu=timing <command line to execute the simulated program>
```

Use a simple matrix multiplication program as an example:

```
./build/ARM/gem5.opt configs/example/arm/starter_se.py --cpu=timing "<path to PerfVec>/Tutorials/mm 8 8 8"
```
