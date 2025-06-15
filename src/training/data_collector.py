class DataCollector:
    """Collects episodes for training"""
    
    def __init__(self, env, sampler):
        self.env = env
        self.sampler = sampler
        
    def collect_episode(self, policy):
        """Collect one episode with given policy"""
        # Sample random episode data
        episode_data = self.sampler.sample_episode()
        
        # Initialize environment with this data
        state = self.env.reset(episode_data)
        
        trajectory = []
        done = False
        
        while not done:
            # Policy makes decision
            action = policy(state)
            
            # Environment step
            next_state, reward, done, info = self.env.step(action)
            
            # Store transition
            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state
            })
            
            state = next_state
            
        return trajectory