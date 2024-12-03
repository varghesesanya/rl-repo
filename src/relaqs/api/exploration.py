from gymnasium import Space
import ray
from ray.rllib.utils.exploration import Exploration
from ray.rllib.policy.policy import Policy
from ray.rllib.env import BaseEnv

class CustomExploration(Exploration):
    def __init__(self, 
        action_space: Space, 
        *, 
        framework: str, 
        policy_config: dict, 
        model: object, 
        num_workers: int, 
        worker_index: int, 
        initial_noise_scale: float = 1.0, 
        noise_decay: float = 0.99, 
        scale_timesteps: int = 10000, 
        **kwargs):
        """
        Custom exploration for DDPG with dynamic noise scaling.
        
        Args:
            action_space: Action space of the environment.
            framework: Framework being used (e.g., "torch", "tf").
            policy_config: RLlib policy configuration (if needed).
            initial_noise_scale: Initial noise scale for exploration.
            noise_decay: Decay rate for the noise scale.
            scale_timesteps: Timesteps over which noise is scaled.
            kwargs: Additional arguments (for future compatibility).
        """
        
        super().__init__(action_space, framework=framework, policy_config=policy_config, 
                         model=model, num_workers=num_workers, worker_index=worker_index)
        
        # Initialize custom attributes
        self.noise_scale = initial_noise_scale
        self.noise_decay = noise_decay
        self.scale_timesteps = scale_timesteps
        self.current_timestep = 0
        self.exploration_steps = 0  # Track the number of exploration steps
        self.max_exploration_steps = 1000  # Limit exploration to 1000 steps
        self.gate_reset = False  # Flag to indicate if the gate has been reset

        
    def on_episode_end(self, policy: Policy, *, environment: BaseEnv = None, episode: int = None, tf_sess=None):
        """Modify exploration behavior at the end of each episode."""
        # Retrieve environment and episode information
        env = environment.get_sub_environments()[0]  # assuming single environment
        
        if env.is_fidelity_consistent(threshold=0.9, steps=10):
            print(f"Gate {env.U_target_key}: Fidelity consistent! Moving to next gate.")            
            # Reset exploration noise when the gate is reset
            self.gate_reset = True
            self.exploration_steps = 0  # Reset exploration step counter

        # else:
        #     # Gradually decay exploration noise
        #     self.noise_scale *= self.noise_decay
        #     policy.exploration.update_parameters({"noise_scale": self.noise_scale})
    
    def add_noise_on_gate_switch(self):
        """Apply full exploration noise when switching gates."""
        print(f"Switching to a new gate, applying full noise scale: {self.noise_scale}")
        # Here we simply reset the noise scale to its initial value
        self.noise_scale = 1.0  # Reset the noise scale when switching gates

    

    # def get_exploration_action(
    #     self, *, action_distribution, explore, **kwargs
    # ):
    #     """
    #     Apply custom exploration logic during action sampling.

    #     Args:
    #         action_distribution: The action distribution predicted by the policy.
    #         timestep: The current timestep (used for exploration scheduling).
    #         explore: Whether exploration is enabled.
    #         policy: The policy object calling this method.
    #         kwargs: Additional arguments (e.g., deterministic sampling).

    #     Returns:
    #         A tuple (explored_action, logp):
    #             - explored_action: The action after applying exploration noise.
    #             - logp: The log probability of the explored action (or None).
    #     """
    #     if not explore:
    #         # If exploration is disabled, return deterministic action.
    #         deterministic_action = action_distribution.deterministic_sample()
    #         return deterministic_action, None

    #     # Sample action and add noise for exploration
    #     sampled_action = action_distribution.sample()
    #     noise = self.noise_scale * self.action_space.sample()
    #     explored_action = sampled_action + noise

    #     # Log probability is optional for deterministic policies like DDPG
    #     return explored_action, None
    
    def get_exploration_action(self, *, action_distribution, explore, **kwargs):
        """
        Apply custom exploration logic during action sampling.
        
        Args:
            action_distribution: The action distribution predicted by the policy.
            explore: Whether exploration is enabled.
            kwargs: Additional arguments (e.g., deterministic sampling).

        Returns:
            A tuple (explored_action, logp):
                - explored_action: The action after applying exploration noise (if applicable).
                - logp: The log probability of the explored action (or None).
        """
        if self.exploration_steps >= self.max_exploration_steps:
            # If exploration steps have reached the max limit, stop exploration
            explore = False

        if not explore:
            # If exploration is disabled, return deterministic action.
            deterministic_action = action_distribution.deterministic_sample()
            return deterministic_action, None

        # Add noise for exploration only if the gate has been reset and we are in the exploration period
        if self.gate_reset and self.exploration_steps < self.max_exploration_steps:
            sampled_action = action_distribution.sample()
            noise = self.noise_scale * self.action_space.sample()  # Sample noise based on current scale
            explored_action = sampled_action + noise
            
            # Increment the exploration step counter
            self.exploration_steps += 1

            # After 1000 steps, stop adding noise (or stop exploration)
            if self.exploration_steps >= self.max_exploration_steps:
                self.gate_reset = False  # Reset the gate switch flag

            return explored_action, None
        
        # If no exploration conditions are met, return the deterministic action
        return action_distribution.deterministic_sample(), None

