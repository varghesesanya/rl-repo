from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
import torch
from  torch.linalg import vector_norm
from typing import Dict, Tuple

class GateSynthesisCallbacks(DefaultCallbacks):
    learning_started = False  # Add a flag to track learning state

    def __init__(self, target_fidelity=0.75, check_steps=10):
        self.target_fidelity = target_fidelity
        self.check_steps = check_steps
        self.steps_since_gate_switch = 0
        self.gate_switch_exploration_steps = 5000
        super().__init__()
        
    def on_train_result(self, *, algorithm, result, **kwargs):
        # Check the number of steps sampled so far
        steps_sampled = result["timesteps_total"]
        learning_threshold = algorithm.config["num_steps_sampled_before_learning_starts"]
        
        if steps_sampled >= learning_threshold and not self.learning_started:
            print(f"Learning started! Steps sampled: {steps_sampled}")
            self.learning_started = True  # Update the flag
            
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        worker.env.episode_id = episode.episode_id
        episode.hist_data["q_values"]= []
        episode.hist_data["grad_gnorm"] = []
        episode.hist_data["average_gradnorm"] =[]
        episode.hist_data["actions"]=[]
        
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        env = base_env.get_sub_environments()[0]
        policy = worker.get_policy()

        # Only check for gate switch if we're past initial training
        if env.is_fidelity_consistent(threshold=self.target_fidelity, steps=self.check_steps):
            print(f"Gate {env.U_target_key}: Fidelity consistent! Moving to next gate.")
            
            # Reset exploration on gate switch
            policy.exploration.scale = 0.8
            self.steps_since_gate_switch = 0
            #decay_factor = 1.0 - (self.steps_since_gate_switch / self.gate_switch_exploration_steps)
            #policy.exploration.scale = max(0.02, decay_factor)
            # self.steps_since_gate_switch += 1
                    
            
            # Switch gate
            env.next_environment()
                
                # # Handle exploration decay only after a gate switch
                # if self.steps_since_gate_switch < self.gate_switch_exploration_steps:
                #     decay_factor = 1.0 - (self.steps_since_gate_switch / self.gate_switch_exploration_steps)
                #     policy.exploration.scale = max(0.02, decay_factor)
                #     self.steps_since_gate_switch += 1
                    
                    
                    
                    
            # target_timestep = env.timesteps_total + 5000  # Adjust exploration duration
            
            
            # Increase exploration when switching gates
            # policy = worker.get_policy("default_policy")
            # policy.exploration.reset_noise(scale=1.0)  # Full exploration on gate switch


            
            # # Access the policy and update exploration explicitly
            # policy = worker.get_policy("default_policy")

            # # Adjust exploration behavior
            # target_timestep = env.timesteps_total + 5000  # Adjust exploration duration
            # policy.num_steps_sampled_before_learning_starts = target_timestep
            # #policy.exploration.scale_timestamp = 2500

            # (Optional) Reset policy's internal exploration object if needed
            #policy.exploration.reset()  # Reset to apply new exploration parameters

            # # Optionally, freeze critic temporarily
            # self.training_on_new_gate = True

            # # Optionally freeze critic for initial steps in the new task
            # for param in self.critic.parameters():
            #     param.requires_grad = False        
           
    # def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
    #     env = base_env.get_sub_environments()[0]
    #     policy = worker.get_policy()
        
    #     if env.is_fidelity_consistent(threshold=0.75, steps=10):
    #         # Reset exploration on gate switch
    #         self.steps_since_switch = 0
    #         policy.exploration.scale = 1.0
    #         env.next_environment()
    #     else:
    #         # Decay exploration
    #         self.steps_since_switch += 1
    #         if self.steps_since_switch < self.exploration_steps_after_switch:
    #             decay = 1.0 - (self.steps_since_switch / self.exploration_steps_after_switch)
    #             policy.exploration.scale = max(0.02, decay)

    #     # Log exploration state
    #     episode.custom_metrics["exploration_scale"] = policy.exploration.scale
    #     episode.custom_metrics["steps_since_switch"] = self.steps_since_switch       
                  
                         
    def on_postprocess_trajectory(
            self,
            *,
            worker: RolloutWorker,
            episode: Episode,
            agent_id: str,
            policy_id: str,
            policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, Tuple[Policy, SampleBatch]],
            **kwargs
        ):
        print("-------------------post processing batch------------------------------------------------")
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1

        model = worker.get_policy("default_policy").model
        policy = worker.get_policy("default_policy")
        input_dict = SampleBatch(obs=torch.Tensor(postprocessed_batch['obs']))
        #------------------------> getting q values <--------------------------------------------------------
        model_out_t, _ = model(input_dict, [], None)
        q_values = model.get_q_values(model_out_t, torch.Tensor(postprocessed_batch['actions']))
        episode.hist_data["q_values"].append(q_values.detach().numpy()[0][0])


        #------------------------> getting gradients <--------------------------------------------------------
        batch = SampleBatch(obs=torch.Tensor(postprocessed_batch['obs']),
            actions=torch.Tensor(postprocessed_batch['actions']),
            new_obs = torch.Tensor(postprocessed_batch['new_obs']),
            rewards=torch.Tensor(postprocessed_batch['rewards']),
            terminateds=torch.Tensor(postprocessed_batch['terminateds']),
            truncateds=torch.Tensor(postprocessed_batch['truncateds']),
            weights= torch.Tensor(postprocessed_batch['weights'])
            )
        gradients = policy.compute_gradients(batch)
        gradients_info = gradients[1]
        NoneType = type(None)
        gradients= [x for x in gradients[0] if not isinstance(x, NoneType)]
        average_grad =0
        for grad in gradients:
            average_grad += vector_norm(grad)
        average_grad = average_grad/(len(gradients))
        episode.hist_data['grad_gnorm'].append(gradients_info['learner_stats']['grad_gnorm'])
        episode.hist_data["average_gradnorm"].append(average_grad.numpy())
        
        #----------------------------> Getting actions <-----------------------------------------------------
        episode.hist_data["actions"].append(postprocessed_batch["actions"].tolist())

    # def on_episode_end(self,
    #     worker: RolloutWorker,
    #     base_env: BaseEnv,
    #     policies: Dict[str, Policy],
    #     episode: Episode,
    #     env_index: int,
    #     **kwargs):
    #     episode.custom_metrics["actions"] = episode.user_data
       
