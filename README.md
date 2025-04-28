# CrazyDeployer

This repository contains an experimental toolchain for training and deploying Deep Reinforcement Learning (DRL) agents on the Crazyflie 2.1 drone. The toolchain is built around the following components:
1. [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/index.html): a unified and modular framework built on top of NVIDIA Isaac Sim, which provides a simulation environment for training and testing DRL agents in a massively parallel manner.
2. [STEdgeAI-Core](https://stedgeai-dc.st.com/assets/embedded-docs/index.html): a tool developed by STMicroelectronics for deploying AI models on STM32 microcontrollers. It provides a set of APIs and tools for converting and optimizing models for deployment on STM32 devices.
3. [Crazyflie 2.1 Firmware](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/): the firmware for the Crazyflie 2.1 drone, which provides the low-level control and communication interfaces for the drone.
4. [Crazyflie PC Client](https://www.bitcraze.io/documentation/repository/crazyflie-clients-python/master/): a Python client for interacting with the Crazyflie 2.1 drone from a PC. It provides a high-level interface for flashing, controlling and visualizing the state of the drone.

## C-Model Generation

The C-model generation process is a crucial step in the deployment of DRL agents on the Crazyflie 2.1 drone. The C-model is a lightweight representation of the trained DRL agent that can be executed on the STM32 microcontroller. The C-model generation can be performed with the following command:

```bash
stedgeai generate -m <model_path> -o <output_dir> -w <workspace_dir> --target stm32f5 --c-api st-ai
```

Where:
- `<model_path>`: the path to the trained DRL agent model in ONNX format.
- `<output_dir>`: the directory where the generated C-model will be saved.
- `<workspace_dir>`: the directory where the STEdgeAI workspace is located.