The aim of this project is to equip the Baby robot with a specific set of skills. Specifically, it has to be able to
1. sense the battery level and switch to foraging mode when it drops below a certain threshold
2. spin around to find the charging station using its camera
3. walk to the charging station and stop there. 
The robot's controller (brain) should be evolved using simulations with our  ARIEL system ([Github Page](https://github.com/ci-group/ariel)), then ported to the real-world Gecko. The success criterion is a real-life demonstration in our Bio-inspired Robotics Lab.
(Baby robot is the product from the publication Real-World Evolution of Robot Morphologies: A Proof of Concept)

The current code space is a combination of ariel framework, and my own project py files. baby_robot.py is the robot's configuration; gait_cmaes.py is the training script for gait control, and rotation_behavior.py is the training script for the rotational behavior. 