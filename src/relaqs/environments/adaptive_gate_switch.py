import numpy as np
class AdaptiveGateSwitch:
    def __init__(self, 
                 base_exploration_rate=0.1,
                 warm_up_steps=1000,
                 critic_freeze_steps=500,
                 min_performance_threshold=0.6):
        self.base_exploration_rate = base_exploration_rate
        self.warm_up_steps = warm_up_steps
        self.critic_freeze_steps = critic_freeze_steps
        self.min_performance_threshold = min_performance_threshold
        self.steps_since_switch = 0
        self.performance_history = []
        
    def get_adaptive_exploration(self):
        """Progressive reduction in exploration based on performance"""
        if self.steps_since_switch < self.critic_freeze_steps:
            # High exploration during critic freeze
            return 0.8 - 0.4 * (self.steps_since_switch / self.critic_freeze_steps)
        elif self.steps_since_switch < self.warm_up_steps:
            # Gradual reduction based on performance
            recent_performance = np.mean(self.performance_history[-10:]) if self.performance_history else 0
            performance_factor = max(0, (recent_performance - self.min_performance_threshold) / 
                                  (1 - self.min_performance_threshold))
            time_factor = (self.warm_up_steps - self.steps_since_switch) / self.warm_up_steps
            return self.base_exploration_rate * (1 + 2 * time_factor * (1 - performance_factor))
        return self.base_exploration_rate

    def should_freeze_critic(self):
        """Determine if critic learning should be frozen"""
        return self.steps_since_switch < self.critic_freeze_steps

    def update_policy(self, policy_config, current_fidelity):
        """Update policy parameters based on current state"""
        self.steps_since_switch += 1
        self.performance_history.append(current_fidelity)
        
        # Trim history to recent performance
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
            
        exploration_rate = self.get_adaptive_exploration()
        
        return {
            'learning_rate': policy_config['lr'] * (0.1 if self.should_freeze_critic() else 1.0),
            'exploration_rate': exploration_rate,
            'critic_frozen': self.should_freeze_critic()
        }

class ScalableGateTraining:
    def __init__(self, 
                 num_iterations=100,
                 gate_switch_threshold=0.9,
                 consistency_window=10,
                 performance_scaling=True):
        self.num_iterations = num_iterations
        self.gate_switch_threshold = gate_switch_threshold
        self.consistency_window = consistency_window
        self.performance_scaling = performance_scaling
        self.gate_performance = {}
        
    def adjust_thresholds(self, gate_key, current_iteration):
        """Dynamically adjust thresholds based on training progress"""
        if not self.performance_scaling:
            return self.gate_switch_threshold
            
        # Scale threshold based on iteration progress
        progress_factor = current_iteration / self.num_iterations
        base_threshold = self.gate_switch_threshold
        
        if gate_key in self.gate_performance:
            # Adjust based on historical performance
            hist_performance = self.gate_performance[gate_key]
            performance_factor = np.mean(hist_performance[-10:]) if hist_performance else 0
            adjusted_threshold = base_threshold * (0.8 + 0.2 * progress_factor)
            return min(adjusted_threshold, base_threshold + 0.05 * performance_factor)
        
        return base_threshold * (0.8 + 0.2 * progress_factor)

    def update_performance(self, gate_key, fidelity):
        """Track performance history per gate"""
        if gate_key not in self.gate_performance:
            self.gate_performance[gate_key] = []
        self.gate_performance[gate_key].append(fidelity)