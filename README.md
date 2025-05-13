# CrazyDeployer

This repository contains an experimental toolchain for training and deploying Deep Reinforcement Learning (DRL) agents on the Crazyflie 2.1 drone. The toolchain is built around the following components:
1. [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/index.html): a unified and modular framework built on top of NVIDIA Isaac Sim, which provides a simulation environment for training and testing DRL agents in a massively parallel manner.
2. [STEdgeAI-Core](https://stedgeai-dc.st.com/assets/embedded-docs/index.html): a tool developed by STMicroelectronics for deploying AI models on STM32 microcontrollers. It provides a set of APIs and tools for converting and optimizing models for deployment on STM32 devices.
3. [Crazyflie 2.1 Firmware](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/): the firmware for the Crazyflie 2.1 drone, which provides the low-level control and communication interfaces for the drone.
4. [Crazyflie PC Client](https://www.bitcraze.io/documentation/repository/crazyflie-clients-python/master/): a Python client for interacting with the Crazyflie 2.1 drone from a PC. It provides a high-level interface for flashing, controlling and visualizing the state of the drone.

## Installation

This repository contains a bunch of external software located under the `vendor` directory and included as submodules. This means that to properly setup the repo you should either cloning it with the following command:
```bash
git clone --recursive https://github.com/mettz/internship.git
```
Or, if you have already cloned it, initialize the submodules with the following command:
```bash
git submodule update --init --recursive
```

## C-Model Generation

The C-model generation process is a crucial step in the deployment of DRL agents on the Crazyflie 2.1 drone. The C-model is a lightweight representation of the trained DRL agent that can be executed on the STM32 microcontroller. The C-model generation can be performed with the following command:

```bash
stedgeai generate -m <model_path> -o <output_dir> --target stm32f4 --c-api st-ai --no-workspace
```

Where:
- `<model_path>`: the path to the trained DRL agent model in ONNX format.
- `<output_dir>`: the directory where the generated C-model will be saved.

## Building & Flashing

To build and flash the controller located under `packages/controller` you can follow the standard crazyflie procedure explained [here](https://github.com/bitcraze/crazyflie-firmware/blob/master/docs/building-and-flashing/build.md).

### Important note about the build

To properly build the controller the compiler needs to get find the ST Core AI Core development headers and static library runtime. Those are usually found within the install location of ST Edge AI Core under `Middlewares/ST/AI/Inc` and `Middlewares/ST/AI/Lib`, respectively. During the build process `make` should be able to find these paths and the appropriate files autonomously given that the ST Edge AI Core post installation steps have been followed has reported [here](https://stedgeai-dc.st.com/assets/embedded-docs/setting_env.html) so that the environment variable `STEDGEAI_CORE_DIR` is set and available in the shell. If this is not the case and the build fails due to missing libraries or headers then issued the following command (or the appropiate one for your shell) and retry:
```bash
export STEDGEAI_CORE_DIR=/path/to/STEdgeAI/version
```