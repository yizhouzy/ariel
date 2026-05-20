def behaviour_tree_controller(vision, battery, gait_net, data):
    """Hard-coded behaviour tree with trained sub-skills."""
    robot_state = get_robot_state(data)
    phase = [2 * np.sin(data.time * 3.0 * 2 * np.pi),
             2 * np.cos(data.time * 3.0 * 2 * np.pi)]
    
    if battery < 0.3:
        # HOMING MODE
        # Use vision to determine turn direction
        centroid_x = vision[5]    # from analyze_sections
        area = vision[6]          # distance proxy
        
        if area < 0.01:
            # Station not visible → spin to search
            turn = 0.5   # constant spin
            speed = 0.3  # slow while searching
        else:
            # Station visible → steer toward it
            turn = -centroid_x * 0.8  # proportional steering
            speed = min(1.0, 0.3 + area * 5)  # faster when closer
        
        gait_input = np.concatenate([robot_state, phase, [turn, speed]])
        return gait_net.forward(gait_input.astype(np.float32))
    else:
        # EXPLORE MODE
        # Random slow wandering
        turn = np.sin(data.time * 0.5) * 0.3  # gentle oscillating turn
        speed = 0.5
        gait_input = np.concatenate([robot_state, phase, [turn, speed]])
        return gait_net.forward(gait_input.astype(np.float32))