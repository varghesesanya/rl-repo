from relaqs.api import gates
from relaqs.environments.gate_synth_env_rllib_Haar import GateSynthEnvRLlibHaarNoisy
from relaqs.plot_data import plot_data, plot_inferencing_data
from relaqs.save_results import SaveResults
from relaqs.api.utils import do_inferencing_new_gate, run_multigate_training, tic, toc
import pandas as pd
import numpy as np
import os

start = tic()

def generate_random_su2_gates(num_gates=10):
    random_gates = []
    for i in range(num_gates):
        random_gate = gates.RandomSU2()  # Creates a random SU2 gate
        random_gates.append(random_gate)
        print(f"Generated random SU2 gate {i+1}")
    
    return random_gates


# Configuration
noise_file = "april/ibmq_belem_month_is_4.json"
inferencing_noise_file = noise_file
n_episodes_for_inferencing = 10
save_training = True
save_inference = True
plot = True
n_training_iterations = 100
path_to_save_checkpoints = "/Users/sanyavarghese/rl-repo/src/policies/checkpoints/new_gate"

# Generate random gates for training
multi_gate_training_list = generate_random_su2_gates(10)
figure_title_training = "Training with 10 Random SU2 Gates"

# Train on the random gates
trained_alg, results = run_multigate_training(
    env_class=GateSynthEnvRLlibHaarNoisy,
    gate=multi_gate_training_list[0],
    n_training_iterations=n_training_iterations,
    noise_file=noise_file,
    path_to_save_checkpoints=path_to_save_checkpoints
)

# Save Training Results
if save_training:
    env = trained_alg.workers.local_worker().env
    sr = SaveResults(env, trained_alg)
    save_dir_training = sr.save_results("train")
    print("Training results saved to:", save_dir_training)
    if plot:
        plot_data(
            save_dir_training,
            episode_length=trained_alg._episode_history[0].episode_length,
            figure_title=figure_title_training,
            env_data_path="env_data_train.csv"
        )
        print("Training plots created.")

# Perform inference on multiple gates
num_inference_gates = 5
inference_gate_indices = np.random.choice(10, num_inference_gates, replace=False)
print(inference_gate_indices)
all_inference_results = []  # To store results for all gates

print("\n=== Starting Inference on Multiple Gates ===")
# For inference on multiple gates
inference_results = []
for idx, gate_index in enumerate(inference_gate_indices):
    print(f"\nInferencing on Gate {gate_index + 1}")
    new_target_gate = multi_gate_training_list[gate_index]
    
    # Run inference
    inferenced_env = do_inferencing_new_gate(
        env, trained_alg, n_episodes_for_inferencing,
        quantum_noise_file_path=inferencing_noise_file,
        new_gate=new_target_gate
    )
    
    # Save results
    if save_inference:
        sr = SaveResults(inferenced_env, trained_alg)
        save_dir_inference = sr.save_results("inference")
        sr.save_inference_summary([new_target_gate])  # Save gate-specific summary
        
        if plot:
            figure_title_inference = f"Inference Results - Gate {gate_index + 1}"
            plot_inferencing_data(
                save_dir_inference,
                episode_length=trained_alg._episode_history[0].episode_length,
                figure_title=figure_title_inference,
                env_data_path="env_data_inference.csv"
            )