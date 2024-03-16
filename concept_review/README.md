# QUIZ 2 -- version 1.0
[ ] 2D/3D coordinate transforms
[ ] Image processing
[ ] Visumotor policy learning
[ ] Forward kinematics
[ ] Training models
[ ] Planning algorithms
[ ] General content from guest lectures.


- [QUIZ 2 -- version 1.0](#quiz-2----version-10)
  - [2D/3D coordinate transforms](#2d3d-coordinate-transforms)
  - [Image processing](#image-processing)
    - [Convolution neural network survival kit](#convolution-neural-network-survival-kit)
    - [Shape Completion with 3D CNN](#shape-completion-with-3d-cnn)
  - [Visumotor policy learning](#visumotor-policy-learning)
    - [Intro](#intro)
      - [Learning for vision v.s. Learning for planning](#learning-for-vision-vs-learning-for-planning)
      - [Challenges in robot learning and overview of solutions](#challenges-in-robot-learning-and-overview-of-solutions)
    - [Imitation learning (BC)](#imitation-learning-bc)
    - [Self-supervised learning](#self-supervised-learning)
      - [What could be a Self-supervision signal?](#what-could-be-a-self-supervision-signal)
      - [Typical Thinking Structure](#typical-thinking-structure)
        - [What's the self-supervision signal?](#whats-the-self-supervision-signal)
        - [How was the data collected?](#how-was-the-data-collected)
        - [How the learning process was configured? (INPUT, OUTPUT, LOSS)](#how-the-learning-process-was-configured-input-output-loss)
    - [Affordance-Based pick and place](#affordance-based-pick-and-place)
      - [Spatial Equivariance!](#spatial-equivariance)
  - [Forward kinematics](#forward-kinematics)
  - [Training models](#training-models)
  - [Planning algorithms](#planning-algorithms)
    - [Foundamental](#foundamental)
    - [Sampling-based planning](#sampling-based-planning)
      - [PRM (Probabilistic Roadmap)](#prm-probabilistic-roadmap)
      - [RRT (Rapidly-exploring Random Tree)](#rrt-rapidly-exploring-random-tree)
  - [General content from guest lectures.](#general-content-from-guest-lectures)


## 2D/3D coordinate transforms

## Image processing
* Lec 8
> U-NET
* Adjust Brightness
  * Add a constant value to each pixel.
  * Multiply each pixel by a constant value.
  * **GAMMA correction**: Non-linear transformation.
    * gamma < 1: reduce contrast + better visibility in dark areas.
* Image Filtering
  * **Convolution Kernel**: Apply a filter to the image.
  * Padding
  * Edge Detection
    * Sobel Operator (Vertical and Horizontal)

### Convolution neural network survival kit
* Basic Layers
  * Convolutional Layer (K (Kernel), N (Kernel Size), S (Stride), P (Padding))
    * INPUT: (C, H, W); OUTPUT: (C', H', W')
      * C' = K
      * H' = (H - N + 2P) / S + 1
      * W' = (W - N + 2P) / S + 1
    * Concepts:
      * **Stride**: How much the filter kernel moves.
      * **Padding**: Add zeros to the border of the image.
      * **Output Size**: (W - F + 2P) / S + 1
  * Pooling Layer
    * **Max Pooling**: Take the maximum value in the filter.
    * **Average Pooling**: Take the average value in the filter.
  * Fully Connected Layer
    * Only used at the end of the network. Map hidden features to the prediction.
      * E.g. Convert 2D IMG to 1D CLASS.
* Loss Function
  * Regression
    * L1 Loss (MAE); L2 Loss (MSE)
  * Classification
    * Cross-Entropy Loss


### Shape Completion with 3D CNN
Input observed surface and output full 3D volume for occluded voxels.

## Visumotor policy learning
* Lec14
### Intro
* Learning-based planning and decision making
#### Learning for vision v.s. Learning for planning
* **Act** --(Cycle time)-- **Sense** --(Perception)-- **Think** --(Planning)-- **Act**
* Two main type of systems:
  * Modularized system: Separate modules for perception, planning, and control.
  * End-to-end system: from raw sensor data to control commands.
    * Hopefully: less engineering, more generalization, and better performance (e.g. robust to sensor noise).
#### Challenges in robot learning and overview of solutions
* Sequential POMDP for robot learning vs SINGLE MDP for vision calssification.
  * Safety Check?
  * State Reset?
  * Task Complexity: Needs a lot of data.
  * Rewards:
    * No supervisor, only a reward signal.
    * Reward is delay and sparse (ONLY REWARD FOR A SEQUENCE OF ACTIONS)

### Imitation learning (BC)
> Human demonstration (obs + act) --> Training data --> Supervised learning --> Policy 
* [CHALLENGE] Model Drifting: The model will drift away from the expert's policy.
  * Triggered by the distribution mismatch between the training and test data (real sensor obs).
  * [Solution] Data Agumentation: Add noise to both the input (OBS) and output (ACT) pair.
    * Instead of training on a single trajectory, train on a distribution of trajectories. 
    * E.g. [DAgger] Dataset Aggregation (Generate new OBS and require require human to label the new ACT).
  * [WHY] Why fails to fit the expert?
    * [DIFFICULTY] Non-Markovain Behavior; (Depends on the history of the state)
      * [Fix] Use 3D CNN
    * [DIFFICULTY] **Multimodal Behavior**; (Multiple actions are good candidates, **direct regression will return the average of the actions which can be significantly bad**.)
      * [Attempt-1] Discrete Action Space: Use classification instead of regression.
        * [ISSUE] Limited precision.
      * [Attempt-2] Mixture of Gaussians as output.
        * [ISSUE] Based on assumption that the expert's policy is a mixture of Gaussians.
      * [Attempt-3] Infer best action with Implicit function: Sample many actions and evaluate their scores before picking the best one (smallest drifting value).
        * From f(x) = 0 to f(x, y) = 0
        * [ISSUE] Need a lot of samples.
        * [SOLUTION] **DIFFUSION POLICY** -- Fits expert more accurately.
          * Reverse stochastic process: Start from the goal and go backward to the initial state.
### Self-supervised learning
> Mainly from Lec 15
* No need for reward / human labels. (no manully anotation)
#### What could be a Self-supervision signal?
* Predict the future.
  * Predict new OBS from the current (OBS, ACT) pair.
#### Typical Thinking Structure
##### What's the self-supervision signal?
##### How was the data collected?
##### How the learning process was configured? (INPUT, OUTPUT, LOSS)

### Affordance-Based pick and place
> Links visual perception of the environment to the robot's possible action.
#### Spatial Equivariance!

## Forward kinematics

## Training models

## Planning algorithms
### Foundamental
* configuration
* degree of freedom
* configuration space (C-space)
  * Wrapped angle (0 ~ 2pi) for C-space obstacle.
### Sampling-based planning
> Define distance in configuration space!   
> Check Collision   
> Define distance with wrapped angle!

#### PRM (Probabilistic Roadmap)
> PRM is not complete, but it is probabilistically complete.
* Two Phases
  * Preprocessing Phase
    * Sample N times
    * Connect the nodes to all valid neighbors.
  * Query Phase
    * In roadmap, find sequence of nodes (milestones) that connects the start and goal.
    * Static roadmap: Not good for dynamic environment.

#### RRT (Rapidly-exploring Random Tree)
> Assume delta is smaller enough to be considered as collision-free.  
> Work well in high-dimensional space.
* Sample -- Steering -- Safety Check -- Add to Tree

* [ADVANCE] Bi-directional RRT
  * Start from both start and goal.
  * Connect the two trees when they are close enough.
## General content from guest lectures.