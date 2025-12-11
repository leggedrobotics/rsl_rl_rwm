# Model-Based RSL RL

A fast and simple implementation of RL algorithms, designed to run fully on GPU.
This code is a fork of [RSL RL](https://github.com/leggedrobotics/rsl_rl) incorporated with model-based RL algorithms supporting [Robotic World Model](https://sites.google.com/view/roboticworldmodel/home).


![Robotic World Model](rwm.png)

**Paper**: [Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics](https://arxiv.org/abs/2501.10100)  
**Project Page**: [https://sites.google.com/view/roboticworldmodel](https://sites.google.com/view/roboticworldmodel)

**Authors**: [Chenhao Li](https://breadli428.github.io/), [Andreas Krause](https://las.inf.ethz.ch/krausea), [Marco Hutter](https://rsl.ethz.ch/the-lab/people/person-detail.MTIxOTEx.TGlzdC8yNDQxLC0xNDI1MTk1NzM1.html)  
**Affiliation**: [ETH AI Center](https://ai.ethz.ch/), [Learning & Adaptive Systems Group](https://las.inf.ethz.ch/) and [Robotic Systems Lab](https://rsl.ethz.ch/), [ETH Zurich](https://ethz.ch/en.html)


## Setup

The package can be installed via PyPI with:

```bash
pip install rsl-rl-lib
```

or by cloning this repository and installing it with:

```bash
git clone https://github.com/leggedrobotics/rsl_rl_rwm.git
cd rsl_rl_rwm
pip install -e .
```

The package supports the following logging frameworks which can be configured through `logger`:

* Tensorboard: https://www.tensorflow.org/tensorboard/
* Weights & Biases: https://wandb.ai/site
* Neptune: https://docs.neptune.ai/

For a demo configuration of PPO, please check the [example_config.yaml](config/example_config.yaml) file.


## Citing

If you use the library with model-based reinforcement learning, please cite the following work:

```text
@article{li2025robotic,
  title={Robotic world model: A neural network simulator for robust policy optimization in robotics},
  author={Li, Chenhao and Krause, Andreas and Hutter, Marco},
  journal={arXiv preprint arXiv:2501.10100},
  year={2025}
}
```
