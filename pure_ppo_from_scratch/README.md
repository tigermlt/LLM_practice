# PPO from scratch
This is a toy project that implements PPO from sratch. It doesn't involve LLM, instead it uses Cart Pole in openAI gym as the environment to give next state, reward etc given an action

The Cart Pole problem can be seen [here](https://gymnasium.farama.org/environments/classic_control/cart_pole/). It has 2 actions and 4 states/observations.

# PPO Overview

phase 1: rollout creation (sample training dataset)

- random initialize the initial state, let model generate action, log probability of this specific action, entropy of the probability distribution of the action space (use as a loss to enforce exploration) and value of the state
- do the action and get next state, reward from the environment
- do the above steps for N trajectories (batch size of N) and T steps (sequence length of T) and populate next state after T which would be `S_{t+1}`, states from step 0 to T, actions from step 0 to T, log probs, values, rewards etc
- with the popualted features, compute advantage function. The advantage function can be computed by traversing from bottom to top with formula:

    Temporal difference (TD) error:  
    `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`

    Advantage estimation (GAE):  
    `A_hat_t = delta_t + gamma * lambda * A_hat_{t+1}`

    Return estimation:  
    `return_t = A_hat_t + V_t`

- create a dataloader for training. Load state, log prob of action, action, advantage, return

phase 2: update model/agent

- the model consists of several Fully connected layer as the shared fondational model. It has two extra head on top of it, one policy head, the other is value head. Use shared layer as policy and value estimation have some overlap.
- Given a state, the policy head will output logits of each action. The logits will pass through softmax to get probability of each action. We can also get the entropy of the actions. 
- Given a state, the value  head will output a single number indicating the estimation of the expected return of the state.
- In the training loop:
1. get one batch from the sampling data (generated using old model)
2. using the new model to compute the log prob of action, entropy and new value given the old state and old action from the sampling data
3. compute policy loss using old log prob of action, new log prob of action, old advantage:
`$L^{CLIP} = min(\frac{\pi_{theta}(a_{t}|s_{t})}{\pi_{theta_{old}}(a_{t}|s_{t})}.\hat{A_{t}}, clip(\frac{\pi_{theta}(a_{t}|s_{t})}{\pi_{theta_{old}}(a_{t}|s_{t})},1-ϵ, 1+ϵ).\hat{A_{t}})$`
Note: the above formula is for maximizing the adv of the minimum of either original adv or cliped adv
Since pytorch minimizes loss, we need to minimize the maximum of either -1 * original adv or -1 * cliped adv
4. compute value loss using MSE bewteen new value and old return
Note: goal is to minimize this loss, so no need to add negative in pytorch
5. compute average of entropy (entropy computed in the new model) in the batch. We need to maximize the average entropy. So minimize -1 * avg_entropy in pytorch.
6. total loss is the combination of policy loss (clipped loss), MSE loss and the entropy loss. Make sure the equivalent final goal is to maximize adv, minimize MSE, maximize entropy.
7. back propagate the loss to update model


# Result
Finally we should see the expected return to go up as we train more steps
![step vs return](img/step_vs_return.png)

