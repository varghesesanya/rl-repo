import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np
import json
from itertools import chain
import matplotlib as mpl
from relaqs import RESULTS_DIR

def plot_results(save_dir, figure_title=""):
    with open(save_dir + "train_results_data.json") as file:  # q values and gradient vector norms
        results = json.load(file)

    q_values = [r['q_values'] for r in results] 
    average_grad_norm = [r["average_gradnorm"] for r in results]

    # Flatten lists
    q_values = list(chain.from_iterable(q_values))
    average_grad_norm = list(chain.from_iterable(average_grad_norm))

    # q values
    rolling_average_window = 100
    q_series = pd.Series(q_values)
    q_windows = q_series.rolling(rolling_average_window)
    q_moving_averages = q_windows.mean().to_list()
    
    # gradient norms
    grad_norm_series = pd.Series(average_grad_norm)
    grad_norm_windows = grad_norm_series.rolling(rolling_average_window)
    grad_norm_moving_averages = grad_norm_windows.mean().to_list()

    # -------------------->  q values <--------------------------
    # -------------------------------> Plotting <-------------------------------------
    rcParams['font.family'] = 'serif'
    mpl.style.use('seaborn-v0_8')

    fig,(ax1, ax2,) = plt.subplots(1, 2) 
    fig.suptitle(figure_title)
    fig.set_size_inches(10, 5)

    ax1.plot(q_values, color="m")
    ax1.plot(q_moving_averages, color="k")
    ax1.set_title("Q Values")
    ax1.set_title("a)", loc='left', fontsize='medium')
    ax1.set_xlabel("Episodes")

    ax2.plot(average_grad_norm, color="slateblue")
    ax2.plot(grad_norm_moving_averages, color="k")
    ax2.set_title("Average Gradient Norms")
    ax2.set_title("b)", loc='left', fontsize='medium')
    ax2.set_xlabel("Episodes")
    plt.tight_layout()
    plt.savefig(save_dir + "gradient_and_q_values.png")


def plot_data(save_dir, episode_length, figure_title='', env_data_path=''):
    """ Currently works for constant episode_length """
    #---------------------- Getting data from files  <--------------------------------------
    df = pd.read_csv(save_dir + env_data_path, header=0)
    fidelities = np.array(df.iloc[:,0])
    rewards = np.array(df.iloc[:,1])
    episode_ids = np.array(df.iloc[:,4])

    print("max fidelity: ", max(fidelities))
    print("max reward: ", max(rewards))

    # --------> Get fidelity, infidelity, and reward from the last step of the episode <--------
    final_fidelity_per_episode = []
    final_infelity_per_episode = []
    sum_of_rewards_per_episode = []

    current_episode_id = episode_ids[0]
    current_fidelity = fidelities[0]
    current_reward_sum = rewards[0]
    for i in range(len(episode_ids)):
        if (episode_ids[i] != current_episode_id) or (i == len(episode_ids) - 1):
            final_fidelity_per_episode.append(current_fidelity)
            final_infelity_per_episode.append(1 - current_fidelity)
            sum_of_rewards_per_episode.append(current_reward_sum)
            current_reward_sum = 0
        current_episode_id = episode_ids[i]
        current_fidelity = fidelities[i]
        current_reward_sum += rewards[i]
    # ------------------------------------------------------------------------------------------

    # ----------------------> Moving average <--------------------------------------
    # Fidelity
    rolling_average_window = 100
    avg_final_fidelity_per_episode = []
    avg_final_infelity_per_episode = []
    avg_sum_of_rewards_per_episode = []
    for i in range (len(final_fidelity_per_episode)):
        start = i - rolling_average_window if (i - rolling_average_window) > 0 else 0
        avg_final_fidelity_per_episode.append(np.mean(final_fidelity_per_episode[start: i + 1]))
        avg_final_infelity_per_episode.append(np.mean(final_infelity_per_episode[start: i + 1]))
        avg_sum_of_rewards_per_episode.append(np.mean(sum_of_rewards_per_episode[start: i + 1]))

    # Round averages to prevent numerical error when plotting
    rounding_precision = 6
    avg_final_fidelity_per_episode = np.round(avg_final_fidelity_per_episode, rounding_precision)
    avg_final_infelity_per_episode = np.round(avg_final_infelity_per_episode, rounding_precision)
    avg_sum_of_rewards_per_episode = np.round(avg_sum_of_rewards_per_episode, rounding_precision)


    if len(avg_final_fidelity_per_episode) >= 100: 
        print("Average final fidelity over last 100 episodes", np.mean(avg_final_fidelity_per_episode[-100:]))

    # -------------------------------> Plotting <-------------------------------------
    rcParams['font.family'] = 'serif'
    mpl.style.use('seaborn-v0_8')

    fig,(ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle(figure_title)
    fig.set_size_inches(10, 5)

    # ----> fidelity <----
    ax1.plot(final_fidelity_per_episode, color="b")
    ax1.plot(avg_final_fidelity_per_episode, color="k")
    ax1.set_title("Fidelity")
    ax1.set_title("a)", loc='left', fontsize='medium')
    ax1.set_xlabel("Episodes")

    # ----> infidelity <----
    ax2.plot(final_infelity_per_episode, color="r")
    ax2.plot(avg_final_infelity_per_episode, color="k")
    ax2.set_yscale("log")
    ax2.set_title("1 - Fidelity (log scale)")
    ax2.set_title("b)", loc='left', fontsize='medium')
    ax2.set_xlabel("Episodes")

    # ----> reward <----
    ax3.plot(sum_of_rewards_per_episode, color="g")
    ax3.plot(avg_sum_of_rewards_per_episode, color="k")
    ax3.set_title("Sum of Rewards")
    ax3.set_title("c)", loc='left', fontsize='medium')
    ax3.set_xlabel("Episodes")
    
    plt.tight_layout()
    plt.savefig(save_dir + "plot.png")

if __name__ == "__main__":
    save_dir = RESULTS_DIR + "2024-02-27_19-31-17_H/"
    plot_data(save_dir, episode_length=2, figure_title="")



import matplotlib.pyplot as plt
import pandas as pd

def plot_training_and_inferencing(
    training_save_dir, 
    inference_save_dir, 
    training_gate_name="Training Gate", 
    inference_gate_name="Inference Gate", 
    figure_title="Training vs Inference", 
    save_path=None
):
    """
    Plot training and inference data side by side with gate names and optionally save the plot as an image.

    Args:
        training_save_dir (str): Directory containing training results.
        inference_save_dir (str): Directory containing inference results.
        training_gate_name (str): Name of the gate used during training.
        inference_gate_name (str): Name of the gate used during inference.
        figure_title (str): Title of the figure.
        save_path (str): Path to save the plot image. If None, the plot will not be saved.
    """
    # Load data
    train_data = pd.read_csv(f"{training_save_dir}/env_data_train.csv")
    inference_data = pd.read_csv(f"{inference_save_dir}/env_data_inference.csv")
    
    # Extract metrics (e.g., fidelity, rewards)
    train_fidelity = train_data['Fidelity'] if 'Fidelity' in train_data.columns else None
    inference_fidelity = inference_data['Fidelity'] if 'Fidelity' in inference_data.columns else None

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    
    # Training plot
    if train_fidelity is not None:
        axs[0].plot(train_fidelity, label=f"Fidelity ({training_gate_name})", color='blue')
        axs[0].set_title(f"Training Results ({training_gate_name})")
        axs[0].set_xlabel("Episodes")
        axs[0].set_ylabel("Fidelity")
        axs[0].legend()
    else:
        axs[0].text(0.5, 0.5, "No training fidelity data", horizontalalignment='center', verticalalignment='center')

    # Inference plot
    if inference_fidelity is not None:
        axs[1].plot(inference_fidelity, label=f"Fidelity ({inference_gate_name})", color='green')
        axs[1].set_title(f"Inference Results ({inference_gate_name})")
        axs[1].set_xlabel("Episodes")
        axs[1].legend()
    else:
        axs[1].text(0.5, 0.5, "No inference fidelity data", horizontalalignment='center', verticalalignment='center')

    # Set the overall title and layout
    fig.suptitle(figure_title)
    plt.tight_layout()
    
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # Show the plot
    plt.show()
