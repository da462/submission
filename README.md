Accompanying code for the ICML 2023 submission titled 'Delayed Feedback in Kernel Bandits': https://dl.acm.org/doi/10.5555/3618408.3619855

We provide code for understanding the regret performance of BPE-Delay, GP-UCB-SDF, BPE or GP-UCB with varying levels of stochastic delayed feedback. We acknowledge the use of the open source code from the paper: https://arxiv.org/pdf/2110.07788.pdf. The code in the refrenced repository contains code for understanding the regret performance of BPE and GP-UCB without delayed feedback. The code for generating the objective function is also from the referenced repository. 

The ```GP-UCB``` class was used to understand the performance of ```GP-UCB-SDF``` when no delay is present.

In order to conduct experiments, follow these steps similar to those in the referenced repsoitory:
* Generate an objective function as described in the referenced repository.
* Run main.py to execute ```BPE```, ```BPE-Delay```, ```GP-UCB-SDF``` or ```GP-UCB```.

```python main.py [function_name] [beta] [poisson_parameter] [algorithm]```
* The supported choice of beta is ```[6]```. Edit the dictionary BETA at the top of main.py to add more choices of beta.
