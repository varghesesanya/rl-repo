import os
import datetime
import numpy as np
import ray
from ray import tune
from ray.air import RunConfig
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.search.optuna import OptunaSearch
from relaqs.api import gates
from relaqs.api.utils import sample_noise_parameters
from relaqs.environments.gate_synth_env_rllib_Haar import GateSynthEnvRLlibHaarNoisy
from relaqs.api.callbacks import GateSynthesisCallbacks
from relaqs import RESULTS_DIR
from qutip.operators import sigmam, sigmaz


def save_hpt_table(results: tune.ResultGrid):
    """Save the hyperparameter tuning results to a CSV file."""
    df = results.get_dataframe()
    path = os.path.join(RESULTS_DIR, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-HPT"))
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path, "hpt_results.csv"))
    print(f"Hyperparameter tuning results saved to: {path}")


def objective(config):
    """
    Objective function for hyperparameter tuning.
    Configures and trains the RLlib algorithm, evaluates fidelity performance.
    """
    # Configure the environment
    env_config = GateSynthEnvRLlibHaarNoisy.get_default_env_config()
    env_config["U_target"] = gates.X().get_matrix()

    # Load quantum noise data
    t1_list, t2_list, detuning_list = sample_noise_parameters()

    env_config["relaxation_rates_list"] = [
        np.reciprocal(t1_list).tolist(),
        np.reciprocal(t2_list).tolist()
    ]
    env_config["detuning_list"] = detuning_list
    env_config["relaxation_ops"] = [sigmam(), sigmaz()]
    env_config["observation_space_size"] = 2 * 16 + 1 + 2 + 1  # Define based on environment specs
    env_config["verbose"] = True

    # Configure the RLlib algorithm
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    alg_config.environment(GateSynthEnvRLlibHaarNoisy, env_config=env_config)
    alg_config.rollouts(batch_mode="complete_episodes")
    alg_config.callbacks(GateSynthesisCallbacks)
    alg_config.train_batch_size = GateSynthEnvRLlibHaarNoisy.get_default_env_config()["steps_per_Haar"]

    # Set hyperparameters from search space
    alg_config.actor_lr = config["actor_lr"]
    alg_config.critic_lr = config["critic_lr"]
    alg_config.actor_hiddens = [config["actor_layer_size"]] * config["actor_num_hiddens"]
    alg_config.critic_hiddens = [config["critic_layer_size"]] * config["critic_num_hiddens"]
    alg_config.exploration_config["scale_timesteps"] = 10000

    # Build and train the algorithm
    alg = alg_config.build()
    for _ in range(config["n_training_iterations"]):
        alg.train()

    # Record performance metrics
    env = alg.workers.local_worker().env
    fidelities = [transition[0] for transition in env.transition_history]
    averaging_window = min(50, len(fidelities))
    avg_final_fidelities = np.mean(fidelities[-averaging_window:])

    return {
        "max_fidelity": max(fidelities, default=0),
        "avg_final_fidelities": avg_final_fidelities,
        "final_fidelity": fidelities[-1] if fidelities else 0,
        "final_reward": env.transition_history[-1][1] if env.transition_history else 0
    }


def run_ray_tune(environment, n_configurations=100, n_training_iterations=50, save=True):
    """
    Runs Ray Tune for hyperparameter tuning.
    """
    ray.init(ignore_reinit_error=True)

    search_space = {
        "actor_lr": tune.loguniform(1e-5, 1e-3),
        "critic_lr": tune.loguniform(1e-5, 1e-3),
        "actor_num_hiddens": tune.choice([1, 2, 3]),
        "actor_layer_size": tune.choice([30, 50, 100, 300]),
        "critic_num_hiddens": tune.choice([1, 2, 3]),
        "critic_layer_size": tune.choice([50, 100, 300, 500]),
        "target_noise": tune.uniform(0.0, 4.0),
        "n_training_iterations": n_training_iterations
    }

    tuner = tune.Tuner(
        objective,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="avg_final_fidelities",
            mode="max",
            search_alg=OptunaSearch(),
            num_samples=n_configurations
        ),
        run_config=RunConfig(
            stop={"training_iteration": n_training_iterations},
        ),
    )
    results = tuner.fit()

    # Get the best configuration
    best_result = results.get_best_result(metric="avg_final_fidelities", mode="max")
    print("Best Configuration:", best_result.config)

    # Save the results
    if save:
        save_hpt_table(results)

    ray.shutdown()


if __name__ == "__main__":
    # Parameters for tuning
    environment = GateSynthEnvRLlibHaarNoisy
    n_configurations = 20
    n_training_iterations = 50
    save_results = True

    # Run the hyperparameter tuning
    run_ray_tune(environment, n_configurations, n_training_iterations, save_results)
