<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/ktechhub/doctoc)*

- [Prerequisite](#prerequisite)
  - [Prerequisite for Steps 1 - 2](#prerequisite-for-steps-1---2)
  - [Prerequisite for Step 3](#prerequisite-for-step-3)
- [Step 1: Prepare a RISC-V cross compiler in Docker](#step-1-prepare-a-risc-v-cross-compiler-in-docker)
  - [Prepare your docker container locally](#prepare-your-docker-container-locally)
- [Step 2: Cross compile HelloWorld source code](#step-2-cross-compile-helloworld-source-code)
- [Step 3: Trace generation (gem5 simulation)](#step-3-trace-generation-gem5-simulation)
  - [3.1 Prepare simulator](#31-prepare-simulator)
    - [3.1.1 Clone ```gem5``` on your server.](#311-clone-gem5-on-your-server)
    - [3.1.2 Build gem5 simulator](#312-build-gem5-simulator)
  - [3.2 Launch gem5 simulation & trace generation](#32-launch-gem5-simulation--trace-generation)
- [Step 4: Model testing](#step-4-model-testing)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Prerequisite
## Prerequisite for Steps 1 - 2
Environment: personal laptop
1. Docker version >= 27.2.0

## Prerequisite for Step 3
Environment: server (with higher computing power for simulation)

# Step 1: Prepare a RISC-V cross compiler in Docker

A RISC-V cross compiler is likely needed for any non-RISC-V machine that you want to use for cross-compiling a RISC-V binary. 
Let’s say you want to use your local laptop for RISC-V compilation, but it is not a RISC-V architecture machine.
Currently, we provide a reproducible and portable Docker container environment as a solution. The Docker image ```khsubnl/riscv_cross_compiler``` from DockerHub is a pre-built image: [https://hub.docker.com/r/khsubnl/riscv_cross_compiler](https://hub.docker.com/r/khsubnl/riscv_cross_compiler) (current size: 8.42 GB)
that contains a pre-compiled riscv cross compiler.

## Prepare your docker container locally

 - Install docker on your laptop.

    - For Mac: [https://docs.docker.com/desktop/setup/install/mac-install/](https://docs.docker.com/desktop/setup/install/mac-install/)

    - For Windows: [https://docs.docker.com/desktop/install/windows-install/](https://docs.docker.com/desktop/install/windows-install/)
   
 - Clone this repo:
   ```
   git clone https://github.com/PerfVec/PerfVec.git
   cd <path to PerfVec>
   ```

 - Use script to setup the docker container:
   ```
   cd scripts/Tutorials/2025_SIGMETRICS
   sh docker_pull_setup.sh 
   ```

 - Run the container in an interactive docker environment:
   ```
   sh docker_run.sh
   ```

You will then be logged into a Docker container environment that contains the RISC-V cross compiler ```riscv64-unknown-linux-gnu-gcc``` under ```/opt/riscv/bin/```.

Simply call ```/opt/riscv/bin/riscv64-unknown-linux-gnu-gcc <any command>``` in this Docker container environment to cross-compile your target program.

Currently the Docker image is based on ubuntu20:04.

# Step 2: Cross compile HelloWorld source code

Now you are within the docker environment. 
Run cross compilation:
```
/opt/riscv/bin/riscv64-unknown-linux-gnu-gcc -static ./hello.c -o hello.riscvbin
```

Now the `hello.riscvbin` executable can be used for gem5 simulation.

# Step 3: Trace generation (gem5 simulation)

## 3.1 Prepare simulator

### 3.1.1 Clone ```gem5``` on your server.
```
git clone https://github.com/lingda-li/gem5.git
cd <Path to gem5>
git checkout riscv
```

### 3.1.2 Build gem5 simulator
```
scons build/RISCV/gem5.opt -j<N>
```
## 3.2 Launch gem5 simulation & trace generation
```
./build/RISCV/gem5.opt --trace_outdir <Output dir> configs/example/riscv/starter_se.py --cpu=minor <Path to executable>
```

For example,
```
./build/RISCV/gem5.opt --trace_outdir ./ configs/example/riscv/starter_se.py --cpu=minor ./hello.riscvbin 
```

There will be two output files, ```trace.txt``` and ```trace.sq.txt```. Refer to [trace.md](../trace.md) for more detail.

# Step 4: Model testing
Continue to the Jupyter Notebook on Colab: [https://colab.research.google.com/drive/1ViJtzsbbUFXkSEYsdgne9iJsfCZq0UvJ#scrollTo=96f_pEaCG1k5](https://colab.research.google.com/drive/1ViJtzsbbUFXkSEYsdgne9iJsfCZq0UvJ#scrollTo=96f_pEaCG1k5) 
for the hands-on activity about model prediction.