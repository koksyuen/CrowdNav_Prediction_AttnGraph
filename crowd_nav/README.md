# Parameters
## config.py
Configuration for:
1. training `training`:
    - `device`: cpu or gpu (cuda)
2. gym environment `env`:
    - `time_limit`: termination state of one episode
    - `time_step`: number of seconds between frames
    - `val_size`: validation
    - `test_size`: testing
    - `randomize_attributes`: randomize human behaviors
    - `num_processes`: depend on numbers of cpu's threads (set in `argument.py`)
    - `record`: record robot states and actions an episode for system identification in **sim2real**
    - `load_act`: 
    - `use_wrapper`: depend on whether is using trajectory prediction model
3. reward function `reward`:
    - `success_reward`: reach goal (termination state of one episode)
    - `collision_penalty`: termination state of one episode
    - `discomfort_dist`: discomfort distance
    - `discomfort_penalty_factor`:
    - `gamma`:
4. simulation `sim`:
    - `circle_radius`: 
    - `arena_size`: 
    - `human_num`: number of pedestrians
    - `human_num_range`: 
    - `predict_steps`: 
    - `predict_method`:
      - `'const_vel'`: constant velocity model
      - `'truth'`: ground truth future trajectory (with info in robot's FOV)
      - `'inferred'`: inferred future traj from GST network
      - `'none'`: no prediction
    - `render`: 
5. other (for save_traj only):
   - `render_traj`
   - `save_slides`
   - `save_path`
6. pedestrians `humans`:
   - `visible`:
   - `policy`: `'orca'` or '`social_force`'
   - `radius`:
   - `v_pref`: preferred velocity
   - `sensor`:
      - `'coordinate'`: 
   - `FOV`: this values * PI
   - `random_goal_changing`: change its goal before it reaches its old goal
   - `goal_change_chance`: possibility of `random_goal_changing`
   - `end_goal_changing`: change its goal after it reaches its old goal
   - `end_goal_change_chance`: possibility of `end_goal_changing`
   - `random_radii`: change its radius after it reaches its current goal
   - `random_v_pref`: change its preferred velocity after it reaches its current goal
   - `random_unobservability`: have a random chance to be blind to other agents at every time step
   - `unobservable_chance`: possibility of `random_unobservability`
   - `random_policy_changing`:
7. robot `robot`:
   - `visible`: whether robot is visible to humans (whether humans respond to the robot's motion)
   - `policy`:
      - `'srnn'`: Decentralized Structural-RNN for Robot Crowd Navigation with Deep Reinforcement Learning
      - `'selfAttn_merge_srnn'`: Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph
   - `radius`:
   - `v_pref`: maximum velocity ??
   - `sensor`:
      - `'coordinate'`: 
   - `FOV`: this values * PI
   - `sensor_range`: radius of perception range
8. action space of robot `action_space`:
   - `kinematics`:
     - `'holonomic'`: vx, vy
     - `'unicycle'`: vx, rotation angle
9. ORCA `orca`:
   - `neighbor_dist`:
   - `safety_space`:
   - `time_horizon`:
   - `time_horizon_obst`:
10. social force `sf`:
    - `A`:
    - `B`:
    - `KI`
11. data collection for trajectory predictor (GST) training `data`:
    - `tot_steps`:
    - `render`:
    - `collect_train_data`:
    - `num_processes`:
    - `data_save_dir`:
    - `pred_timestep`: number of seconds between each position in traj pred model
12. trajectory predictor `pred`:
    - `model_dir`: 
13. LIDAR `lidar`:
    - `angular_res`: angular resolution (offset angle between neighboring rays) in degrees
    - `range`: range in meters
14. simulation to real world `sim2real`:
    - `use_dummy_detect`: use dummy robot and human states or not
    - `record`:
    - `load_act`:
    - `ROSStepInterval`:
    - `fixed_time_interval`:
    - `use_fixed_time_interval`:

## arguments.py
### Training
1. `output_dir`: the saving directory for train.py (RL policy)
2. `resume`: resume training from an existing checkpoint or not
3. `load-path`: path of weights for resume training
4. `overwrite`: whether to overwrite the output directory in training
5. `num_threads`: number of threads used for intraop parallelism on CPU
6. `phase`: only implement in testing
7. `cuda-deterministic`: sets flags for determinism when using CUDA (potentially slow!)
8. `no-cuda`: disables CUDA training
9. `cuda`: True if `torch.cuda.is_available()` and `no-cuda==False`
10. `seed`: random seed (default: 1)
11. `num-processes`: how many training processes to use (depend on number of cpu's threads)
12. `num-mini-batch`: number of batches for PPO training
13. `num-steps`: number of forward steps in A2C training
14. `recurrent-policy`: use a recurrent policy (only support `a2c` and `ppo`, not support `acktr`)
15. `ppo-epoch`: number of ppo epochs
16. `clip-param`: ppo clip parameter
17. `value-loss-coef`: value loss coefficient
18. `entropy-coef`: entropy term coefficient
19. `lr`: learning rate
20. `eps`: RMSprop optimizer epsilon
21. `alpha`: RMSprop optimizer apha
22. `gamma`: discount factor for rewards
23. `max-grad-norm`: max norm of gradients
24. `num-env-steps`: number of environment steps (episode) to train
25. `use-linear-lr-decay`: use a linear schedule on the learning rate
26. `algo`: RL algorithm (`a2c` or `ppo` or `acktr`)
27. `save-interval`: one save per n updates
28. `use-gae`: use generalized advantage estimation
29. `gae-lambda`: gae lambda parameter
30. `log-interval`: one log per n updates
31. `use-proper-time-limits`: compute returns taking into account time limits
### For SRNN only (old paper)
1. `human_node_rnn_size`: size of Human Node RNN hidden state
2. `human_human_edge_rnn_size`: size of Human-Human Edge RNN hidden state
3. `aux-loss`: auxiliary loss on human nodes outputs
4. `human_node_input_size`: dimension of the node features
5. `human_human_edge_input_size`: dimension of the edge features
6. `human_node_output_size`: dimension of the node output
7. `human_node_embedding_size`: embedding size of node features
8. `human_human_edge_embedding_size`: embedding size of edge features
9. `attention_size`: attention vector dimension
10. `seq_length`: sequence length
11. `use_self_attn`: human-human attention network will be included if set to True, else there will be no human-human attention.
12. `use_hr_attn`: robot-human attention network will be included if set to True, else there will be no robot-human attention.
### Simulator
1. `env-name`:
    - `'CrowdSimVarNum-v0'`: no prediction (ORCA, Social Force, SRNN) 
    - `'CrowdSimPred-v0'`: use the ground truth predictor or constant velocity predictor 
    - `'CrowdSimPredRealGST-v0'`: use GST trajectory predictor
### Other
1. `sort_humans`: sort all humans and squeeze them to the front or not