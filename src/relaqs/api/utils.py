import ray
import numpy as np
import pandas as pd
from numpy.linalg import eigvalsh
from scipy.linalg import sqrtm
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ddpg import DDPGConfig
from relaqs import RESULTS_DIR
from relaqs.quantum_noise_data.get_data import (get_month_of_all_qubit_data, get_single_qubit_detuning)
from relaqs.api.callbacks import GateSynthesisCallbacks
from relaqs import QUANTUM_NOISE_DATA_DIR
from qutip.operators import *

vec = lambda X : X.reshape(-1, 1, order="F") # vectorization operation, column-order. X is a numpy array.
vec_inverse = lambda X : X.reshape(int(np.sqrt(X.shape[0])),
                                   int(np.sqrt(X.shape[0])),
                                   order="F") # inverse vectorization operation, column-order. X is a numpy array.

def normalize(quantity, list_of_values):
    """ normalize quantity to [0, 1] range based on list of values """
    return (quantity - min(list_of_values) + 1E-15) / (max(list_of_values) - min(list_of_values) + 1E-15)

def polar_vec_to_complex_matrix(vec, return_flat=False):
    """ 
    The intended use of this function is to convert from the representation of the unitary
    in the agent's observation back to the unitary matrxi.

    Converts a vector of polar coordinates to a unitary matrix. 
    
    The vector is of the form: [r1, phi1, r2, phi2, ...]
    
    And the matrix is then: [-1 * r1 * exp(i * phi1 * 2pi),...] """
    # Convert polar coordinates to complex numbers
    complex_data = []
    for i in range(0, len(vec), 2):
        r = vec[i]
        phi = vec[i+1]
        z = -1 * r * np.exp(1j * phi * 2*np.pi) 
        complex_data.append(z)

    # Reshape into square matrix
    if not return_flat:
        matrix_dimension = int(np.sqrt(len(vec)))
        complex_data = np.array(complex_data).reshape((matrix_dimension, matrix_dimension))

    return complex_data

def superoperator_evolution(superop, dm):
    return vec_inverse(superop @ vec(dm))

def load_pickled_env_data(data_path):
    df = pd.read_pickle(data_path)
    return df

gate_fidelity = lambda U, V: float(np.abs(np.trace(U.conjugate().transpose() @ V))) / (U.shape[0])

def dm_fidelity(rho, sigma):
    assert np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))).imag < 1e-8, f"Non-negligable imaginary component {np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))).imag}"
    #return np.abs(np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))))**2
    return np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))).real**2

def sample_noise_parameters(t1_t2_noise_file=None, detuning_noise_file=None):
    # ---------------------> Get quantum noise data <-------------------------
    if t1_t2_noise_file is None:
        t1_list = np.random.uniform(40e-6, 200e-6, 100)
        t2_list = np.random.uniform(40e-6, 200e-6, 100)
    else:
        t1_list, t2_list = get_month_of_all_qubit_data(QUANTUM_NOISE_DATA_DIR + t1_t2_noise_file) # in seconds

    if detuning_noise_file is None:
        mean = 0
        std = 1e4
        sample_size = 100
        samples = np.random.normal(mean, std, sample_size)
        detunings = samples.tolist()
    else:
        detunings = get_single_qubit_detuning(QUANTUM_NOISE_DATA_DIR + detuning_noise_file)

    return list(t1_list), list(t2_list), detunings

def do_inferencing(alg, n_episodes_for_inferencing, quantum_noise_file_path):
    """
    alg: The trained model
    n_episodes_for_inferencing: Number of episodes to do during the training
    """
    
    assert n_episodes_for_inferencing > 0
    env = return_env_from_alg(alg)
    obs, info = env.reset()
    t1_list, t2_list, detuning_list = sample_noise_parameters(quantum_noise_file_path)
    env.relaxation_rates_list = [np.reciprocal(t1_list).tolist(), np.reciprocal(t2_list).tolist()]
    env.detuning_list = detuning_list
    num_episodes = 0
    episode_reward = 0.0
    print("Inferencing is starting ....")
    while num_episodes < n_episodes_for_inferencing:
        print("episode : ", num_episodes)
        # Compute an action (`a`).
        a = alg.compute_single_action(
            observation=obs,
            policy_id="default_policy",  # <- default value
        )
        # Send the computed action `a` to the env.
        obs, reward, done, truncated, _ = env.step(a)
        episode_reward += reward
        # Is the episode `done`? -> Reset.
        if done:
            print(f"Episode done: Total reward = {episode_reward}")
            obs, info = env.reset()
            num_episodes += 1
            episode_reward = 0.0
    return env, alg

def load_model(path):
    "path (str): Path to the file usually beginning with the word 'checkpoint' " 
    loaded_model = Algorithm.from_checkpoint(path)
    return loaded_model

def get_best_episode_information(filename):
    df = pd.read_csv(filename, names=['Fidelity', 'Reward', 'Actions', 'Flattened U', 'Episode Id'], header=0)
    fidelity = df.iloc[:, 0]
    max_fidelity_idx = fidelity.argmax()
    fidelity = df.iloc[max_fidelity_idx, 0]
    episode = df.iloc[max_fidelity_idx, 4]
    best_episode = df[df["Episode Id"] == episode]
    return best_episode

def get_best_actions(filename):
    best_episode = get_best_episode_information(filename)
    action_str_array = best_episode['Actions'].to_numpy()

    best_actions = []
    for actions_str in action_str_array:
        # Remove the brackets and split the string by spaces
        str_values = actions_str.strip('[]').split()

        # Convert the string values to float
        float_values = [float(value) for value in str_values]

        # Convert the list to a numpy array (optional)
        best_actions.append(float_values)
    return best_actions, best_episode['Fidelity'].to_numpy() 

def run(env_class, gate, n_training_iterations=1, noise_file=""):
    """Args
       gate (Gate type):
       n_training_iterations (int)
       noise_file (str):
    Returns
      alg (rllib.algorithms.algorithm)

    """
    ray.init()
    env_config = env_class.get_default_env_config()
    env_config["U_target"] = gate.get_matrix()

    # ---------------------> Get quantum noise data <-------------------------
    t1_list, t2_list, detuning_list = sample_noise_parameters(noise_file)

    env_config["relaxation_rates_list"] = [np.reciprocal(t1_list).tolist(), np.reciprocal(t2_list).tolist()] # using real T1 data
    env_config["detuning_list"] = detuning_list
    env_config["relaxation_ops"] = [sigmam(),sigmaz()]
    env_config["observation_space_size"] = 2*16 + 1 + 2 + 1 # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 2 for relaxation rate + 1 for detuning
    env_config["verbose"] = True

    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    alg_config.environment(env_class, env_config=env_config)
    alg_config.rollouts(batch_mode="complete_episodes")
    alg_config.callbacks(GateSynthesisCallbacks)
    alg_config.train_batch_size = env_class.get_default_env_config()["steps_per_Haar"]

    ### working 1-3 sets
    alg_config.actor_lr = 4e-5
    alg_config.critic_lr = 5e-4

    alg_config.actor_hidden_activation = "relu"
    alg_config.critic_hidden_activation = "relu"
    alg_config.num_steps_sampled_before_learning_starts = 1000
    alg_config.actor_hiddens = [30,30,30]
    alg_config.exploration_config["scale_timesteps"] = 10000

    alg = alg_config.build()
    list_of_results = []
    for _ in range(n_training_iterations):
        result = alg.train()
        list_of_results.append(result['hist_stats'])

    ray.shutdown()

    return alg

def return_env_from_alg(alg):
    env = alg.workers.local_worker().env
    return env

def load_and_analyze_best_unitary(data_path, U_target):
    df = pd.read_csv(data_path, names=['Fidelity', 'Reward', 'Actions', 'Flattened U', 'Episode Id'], header=0)
    
    fidelity = df["Fidelity"]
    max_fidelity_idx = fidelity.argmax()
    best_flattened_unitary = eval(df.iloc[max_fidelity_idx, 3])

    best_fidelity_unitary = np.array([complex(x) for x in best_flattened_unitary]).reshape(4, 4)
    max_fidelity = fidelity.iloc[max_fidelity_idx]

    print("Max fidelity:", max_fidelity)
    print("Max unitary:", best_fidelity_unitary)

    zero = np.array([1, 0]).reshape(-1, 1)
    zero_dm = zero @ zero.T.conjugate()
    zero_dm_flat = zero_dm.reshape(-1, 1)

    dm = best_fidelity_unitary @ zero_dm_flat
    dm = dm.reshape(2, 2)
    print("Density Matrix:\n", dm)

    # Check trace = 1
    dm_diagonal = np.diagonal(dm)
    print("diagonal:", dm_diagonal)
    trace = sum(np.diagonal(dm))
    print("trace:", trace)

    # # Check that all eigenvalues are positive
    eigenvalues = eigvalsh(dm)
    print("eigenvalues:", eigenvalues)
    #assert (0 <= eigenvalues).all()

    psi = U_target.get_matrix() @ zero
    true_dm = psi @ psi.T.conjugate()
    print("true dm\n:", true_dm)

    print("Density matrix fidelity:", dm_fidelity(true_dm, dm))
