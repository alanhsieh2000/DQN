# Function Approximation
In MDP problems with large number of states, methods which need to estimate action-value of all state-action pairs, called tabular methods, 
are impractical. Value function approximation (VFA) with parameterized form, $\^{v}(s_t, w), \^{q}(s_t, a_t, w)$, is more suitable in this case. 
Among many function approximators, neural network is the most promising one. Because it is differentiable, stochastic gradient decent optimizers 
that are provided in deep learning frameworks like PyTorch and TensorFlow can be used to minimize the loss function

$${\begin{align}
  J(w) &= E_\pi[(v_\pi(S) - \^{v}(S, w))^2], \text{or} \notag &\\
  J(w) &= E_\pi[(q_\pi(S, A) - \^{q}(S, A, w))^2] \notag &\\
\end{align}}$$

For gradient decent minimizers, we want to change $w$ in the opposite direction of $\bigtriangledown J(w)$ so that $J(w + \Delta w)$ is less 
than $J(w)$. We can set 

$${\begin{align}
   \Delta w &= - \frac{1}{2} \alpha \cdot \bigtriangledown J(w) \notag &\\
   &= \alpha \cdot E_\pi[(q_\pi(S, A) - \^{q}(S, A, w)) \cdot \bigtriangledown \^{q}(S, A, w)] \notag &\\
\end{align}}$$

For MC, the estimation of $q_\pi(S_t, A_t)$ is $G_t$, so we have

$${\Delta w = \alpha \cdot E_\pi[(G_t - \^{q}(S_t, A_t, w)) \cdot \bigtriangledown \^{q}(S_t, A_t, w)]}$$

For TD(0), the estimation of $q_\pi(S_t, A_t)$ is $R_{t+1} + \gamma \cdot Q(S_{t+1}, A_{t+1})$, so we have 

$${\Delta w = \alpha \cdot E_\pi[(R_{t+1} + \gamma \cdot \^{q}(S_{t+1}, A_{t+1}, w) - \^{q}(S_t, A_t, w)) \cdot \bigtriangledown \^{q}(S_t, A_t, w)]}$$

For Q-learning, the estimation of $q_\pi(S_t, A_t)$ is $R_{t+1} + \gamma \cdot \max_a Q(S_{t+1}, a)$, so we have 

$${\Delta w = \alpha \cdot E_\pi[(R_{t+1} + \gamma \cdot \max_a \^{q}(S_{t+1}, a, w) - \^{q}(S_t, A_t, w)) \cdot \bigtriangledown \^{q}(S_t, A_t, w)]}$$

# Deep Q-learning
Deep Q-learning (DQN) is Q-learning with function approximation, using a deep neural network as its function approximator (Mnih, 2013). To make 
the algorithm stable, DQN use

1. image preprocessing
    - Raw Atari frames are 210 x 160 pixel images with 128 color palette. State $s_t$ is the last 4 frames.
    - RGB colors are converted to grey-scale.
    - The resolution is down-sampling to 110 x 84 pixel.
    - The image is cropped to 84 x 84 pixel covering the playing area.
    - Pixels are substracted by pixel-level mean and scaled within [-1, 1]. This, $\phi(s_t)$, is the input to the CNN Q function approximator.
2. convolutional neural network
    - The input tensor, $\phi(s_t)$, is 84 x 84 x 4 for the last 4 frames.
    - The first layer is CONV, 8 x 8 x 16 with stride 4 and zero-padding, which outputs a 20 x 20 x 16 tensor. The activation function is 
    $tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$.
    - The second layer is CONV, 4 x 4 x 32 with stride 2 and zero-padding, which outputs a 9 x 9 x 32 tensor. The activation function is also 
    $tanh(x)$.
    - The third layer is Linear, fully connected 256 units with flatten 2592 inputs, which outputs a 256 tensor. The activation function is 
    $ReLU(x) = max(0, x)$.
    - The output layer is Linear, fully connected to 1 output for regression or 4 ~ 18 outputs feeding into a softmax function for classification.
3. experience replay
    - The last $N$ experience tuples, $(\phi(s_t), a_t, r_{t+1}, \phi(s_{t+1}))$ that come from one or more episodes, are stored into a replay memory 
    $D$.
    - A mini-batch of size 32 is uniformly random sampled from $D$.
    - The value of rewards, $r_{t+1}$, is clipped to 1 if the original reward of that game is positive, -1 if it is negative, and 0 otherwise.
    - N is one million frames, 250,000 experience tuples.
4. two parameter copies
    - The copy of learnable parameter $w$ is used by $\^{q}(S_t, A_t, w)$.
    - The copy of non-learnable parameter $w^\prime$ is ued by $\^{q}(S_{t+1}, a, w^\prime)$.
    - In practical, the value of $w^\prime$ is replaced the value of $w$ every 50 training steps.
5. training
    - RMSProp is used.
    - The behavior policy is $\epsilon$-greedy.
    - $\epsilon$ is going down linearly from 1 to 0.1 over the first million frames, 250,000 training steps.
    - $\epsilon$ stays at 0.1 constantly after one million frames.
    - It trains for 10 million frames in total, 2,500,000 training steps.
    - The agent takes actions every $k^th$ frame, where k=3 for Space Invaders and k=4 for the other games.
    - The learning rate is 2e-4, and $\gamma$ is 0.99.
    - The loss function is

$${\begin{align}
  J(w) &= \frac{1}{N} \cdot \sum_{i=0}^{N-1}(y_i - \^{q}(s_{i,t}, a_{i,t}, w))^2, \notag &\\
  &\text{where } y_i = \begin{cases}
    r_{i,t+1}, &\text{ if } s_{i,t+1} = s_T, \notag &\\
    r_{i,t+1} + \gamma \cdot \max_a \^{q}(s_{i,t+1}, a, w^\prime), &\text{ otherwise} \notag &\\
  \end{cases}
\end{align}}$$

Here is the algorithm

1. Initialize $D$ and burn in with $N$ experience tuples by random policy
2. Initialize $\^{q}_w$ and its clone $\^{q}_{w^\prime}$.
3. Initialize c = 0
4. Initialize behavior policy $b(s_t)$ as $\epsilon$-greedy with $\^{q}_w$ and the initial $\epsilon$
4. repeat for training episodes
    - Initialize $s_t = s_0$
    - $\phi_0 = \phi(s_0)$
    - while $s_t \neq s_T$
        - $a_t = b(s_t)$
        - take action $a_t$ and observe $r_{t+1}, s_{t+1}$
        - store $(\phi(s_t), a_t, r_{t+1}, \phi(s_{t+1}))$ in $D$
        - sample a mini-batch with size N from $D$
        - calculate $y_i$ for all samples in the mini-batch, using $\^{q}_{w^\prime}$
        - update $\^q_w$ using $RMSProp(\bigtriangledown J(w))$
        - c = c + 1
        - replace $w^\prime$ with $w$ if c % target_update = 0
        - $s_t = s_{t+1}$
        - update $\epsilon$ of $b(s_t)$ if needed

# Deep Learning for offline MCTS
Deep learning for offline MCTS (Guo, 2014) used the same CNN that used in DQN as its Q function approximator. The difference comes from the 
training approach. The idea is to train the CNN by using supervised learning. The output of DQN's CNN is action-value. In other words, it is 
a regression model. This paper also found that, by replacing the output layer of that CNN with 4 ~ 18 outputs feeding to a softmax function, 
the resulting classification model could be a policy approximator that outperformed the regression model.

The training data came from a UCT agent playing games offline. By playing the game 800 times from start to finish using the UCT agent, it built 
a table as the dataset for training. On each row, there were last 4 frames of each state along each episode and the action-value computed by 
the UCT agent. Another column of the table was the choice of action that was best according to the UCT agent. This column was used as labels 
for training the classification model, $(x_i, y_i) = (\phi(s_{i,t}), a_t)$

The paper also found that an interleaved training protocol for the classification model delivered the best performance.

1. Play the game 200 times from start to finish using the UCT agent and build the table, last 4 frames of each state along each episode and 
the action choosen by the UCT agent on each row.
2. Train the CNN using the data collected in the first step.
3. Play the game 200 times like the first step, but use the trained CNN in the step 2 to choose action as the $\epsilon$-greedy behavior policy 
with 0.05 $\epsilon$. The data to be collected is exactly the same as the first step, last 4 frames of each state along each episode and the 
action choosen by the UCT agent on each row.
4. Re-train the CNN using the data collected so far, from the 400 episodes.
5. Repeat step 3 until 800 episodes accumulated.
6. Re-train the CNN using the final dataset, 800 episodes.

# Reference
- Carnegie Mellon University, Fragkiadaki, Katerina, et al. 2024. "10-403 Deep Reinforcement Learning" As of 8 November, 2024. https://cmudeeprl.github.io/403website_s24/.
- Sutton, Richard S., and Barto, Andrew G. 2018. Reinforcement Learning - An indroduction, second edition. The MIT Press.
- Mnih, Volodymyr, et al. 2013. Playing Atari with Deep Reinforcement Learning, [arXiv:1312.5602v1](https://arxiv.org/abs/1312.5602v1)
- Guo, X., et al. 2014. Deep learning for real-time Atari game play using offline Monte-Carlo tree search planning, Advances in Neural Information Processing Systems.