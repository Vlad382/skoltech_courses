## Final Grade: A (97.00%)

### Homework 1. 11/11 pts

### Homework 2. 19/19 pts

### Homework 3. 10/10 pts

### Final Project. 37/40 pts

[GitHub repo of the project](https://github.com/Vlad382/dqn_with_synced_target_net)
[WandB](https://wandb.ai/skoltech_ml2022_project_synced_target_nets/project/reports/Project-summary--VmlldzoxNzIzOTk0)

Research on the usage of target networks in Deep Reinforcement Learning

*Larger sizes of neural networks in DQN models positively impact the stability of the algorithm. So, hypothesis is that at some point using the target network will be pointless.*

**Results**

 - For DQN and Double DQN models the necessity of target network was proven
 - Direct correlation between the neural network size and stability for Q-learning models without target network was observed: large size of action networks increases stability
 - It was shown, that Dueling DQN model without target network can achieve the same performance as original Dueling DQN