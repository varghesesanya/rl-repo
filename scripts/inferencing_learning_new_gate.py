from relaqs.api import gates
from relaqs.environments.gate_synth_env_rllib_Haar import GateSynthEnvRLlibHaarNoisy
from relaqs.environments.noisy_single_qubit_env import NoisySingleQubitEnv
from relaqs.environments.single_qubit_env import SingleQubitEnv
from relaqs.plot_data import plot_data, plot_training_and_inferencing
from relaqs.save_results import SaveResults
from relaqs.api.utils import do_inferencing_new_gate, run, tic, toc
from ray.rllib.algorithms.algorithm import Algorithm
from relaqs.visualization.bloch_sphere import load_and_plot_noiseless

start = tic()

# Configuration
noise_file = "april/ibmq_belem_month_is_4.json"
inferencing_noise_file = noise_file
n_episodes_for_inferencing = 1500
save_training = True
save_inference = True
plot = True
figure_title_training = "Training with Target Gate X"
figure_title_inference = "Inferencing with Target Gate Y"
n_training_iterations = 50
path_to_save_checkpoints = "/Users/sanyavarghese/rl-repo/src/policies/checkpoints/new_gate"


#-----------------------> Training model <------------------------
# try:
#     trained_alg = Algorithm.from_checkpoint(path_to_save_checkpoints)
#     print("Model loaded successfully for inference.")
    
#     # Check iterations completed
#     completed_iterations = trained_alg.iteration
#     print(f"Completed iterations: {completed_iterations}")
    
#     if completed_iterations < n_training_iterations:
#         print(f"Resuming training for {n_training_iterations - completed_iterations} iterations...")
#         trained_alg = run(
#             GateSynthEnvRLlibHaarNoisy,
#             gates.X(),
#             n_training_iterations=n_training_iterations - completed_iterations,
#             noise_file=noise_file,
#             path_to_save_checkpoints=path_to_save_checkpoints
#         )
#     else:
#         print("Training already completed for the set iterations.")
# except ValueError:
#     print("No saved model found. Training a new model...")
trained_alg = run(
    GateSynthEnvRLlibHaarNoisy,
    gates.X(),
    n_training_iterations=n_training_iterations,
    noise_file=noise_file,
    path_to_save_checkpoints=path_to_save_checkpoints
)

# -------------------> Save Training Results <---------------------------------------

if save_training is True:
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


# ------------------> Inferencing on New Gate <------------------------
# Configure the new target gate for inferencing
new_target_gate = gates.Y() 
inferenced_env = do_inferencing_new_gate(env,
    trained_alg,
    n_episodes_for_inferencing,
    quantum_noise_file_path=inferencing_noise_file,
    new_gate=new_target_gate
)


# ------------------> Save Inferencing Results <------------------------
if save_inference:
    sr = SaveResults(inferenced_env, trained_alg)
    save_dir_inference = sr.save_results("inference")
    print("Inferencing results saved to:", save_dir_inference)
    
    if plot:
        plot_data(
        save_dir_inference,
        episode_length=trained_alg._episode_history[0].episode_length,
        figure_title=figure_title_inference,
        env_data_path="env_data_inference.csv"
        )
        print("Inference plots created.")

# Define the gate names and save path
training_gate_name = "X Gate"
inference_gate_name = "Y Gate"
plot_save_path = "./training_vs_inference_fidelity_with_gate_names.png"

# Plot training vs inference and save the plot
plot_training_and_inferencing(
    training_save_dir=save_dir_training,       
    inference_save_dir=save_dir_inference,  
    training_gate_name=training_gate_name,  
    inference_gate_name=inference_gate_name, 
    figure_title="Training vs Inference Fidelity",
    save_path=plot_save_path            # Path to save the plot
)

print(f"Elapsed Time: {toc(start):.4f} seconds")