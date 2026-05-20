"""A simple strategy of implementing a battery that drain linearly over time. 
- Time = duration of an episode
- If the duration is 30 secs, then the battery will drain from 1.0 to 0.0 in 30 seconds.
"""

# Hyperparameter
duration: float  # duration per episode

# Parameter
BATTERY_THRESHOLD = 0.3  # threshold for the homing behavior
battery = 1.0

# In some way, feed the battery into the network
# My approach:
state_input = np.concatenate([
    robot_state,
    vision_inputs,
    ...
    battery # a float number from 0 to 1
])

# In the fitness function, reward the robot when battery is below threshold
if battery <= BATTERY_THRESHOLD:       # 0.3
    fitness +=... # my approach is distance

# For each physical step (in the main training loop)
dt = model.opt.timestep  # model = mujoco model
drain_per_step = dt / duration # drain per timestep 
battery = max(0.0, battery - drain_per_step)

    
    
