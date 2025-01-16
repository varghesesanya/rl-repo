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
    gate_switch_steps = np.where(df['Gate Switch'] == 1)[0]//2  # Indices of gate switches

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
    rolling_average_window = 10
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
    
     # Add vertical lines for gate switches
    for step in gate_switch_steps:
        ax1.axvline(x=step, color='orange', linestyle='--', label='Gate Switch' if step == gate_switch_steps[0] else "")

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

import pandas as pd
import numpy as np
import plotly.graph_objects as go

def plot_data_using_plotly(save_dir, episode_length, figure_title='', env_data_path=''):
    """Visualize all data points using Plotly."""
    # ---------------------- Getting data from files ----------------------
    df = pd.read_csv(save_dir + env_data_path, header=0)
    print("Length of the dataframe::{}", len(df))
    fidelities = np.array(df.iloc[:, 0])
    rewards = np.array(df.iloc[:, 1])
    episode_ids = np.array(df.iloc[:, 4])
    gate_switch_steps = np.where(df['Gate Switch'] == 1)[0]//2  # Indices of gate switches

    print("Max fidelity:", max(fidelities))
    print("Max reward:", max(rewards))

    # --------> Process fidelity, infidelity, and rewards <--------
    final_fidelity_per_episode = []
    final_infidelity_per_episode = []
    sum_of_rewards_per_episode = []

    current_episode_id = episode_ids[0]
    current_fidelity = fidelities[0]
    current_reward_sum = rewards[0]

    for i in range(len(episode_ids)):
        if (episode_ids[i] != current_episode_id) or (i == len(episode_ids) - 1):
            final_fidelity_per_episode.append(current_fidelity)
            final_infidelity_per_episode.append(1 - current_fidelity)
            sum_of_rewards_per_episode.append(current_reward_sum)
            current_reward_sum = 0
        current_episode_id = episode_ids[i]
        current_fidelity = fidelities[i]
        current_reward_sum += rewards[i]

    # ---------------------- Plotting with Plotly ----------------------
    fig = go.Figure()

    # Fidelity
    fig.add_trace(go.Scatter(
        x=list(range(len(final_fidelity_per_episode))),
        y=final_fidelity_per_episode,
        mode='lines+markers',
        name='Fidelity',
        line=dict(color='blue'),
        marker=dict(size=5)
    ))

    # Infidelity
    fig.add_trace(go.Scatter(
        x=list(range(len(final_infidelity_per_episode))),
        y=final_infidelity_per_episode,
        mode='lines+markers',
        name='Infidelity (1 - Fidelity)',
        line=dict(color='red'),
        marker=dict(size=5),
        yaxis='y2'
    ))

    # Rewards
    fig.add_trace(go.Scatter(
        x=list(range(len(sum_of_rewards_per_episode))),
        y=sum_of_rewards_per_episode,
        mode='lines+markers',
        name='Sum of Rewards',
        line=dict(color='green'),
        marker=dict(size=5)
    ))

    # Add vertical lines for gate switches
    for step in gate_switch_steps:
        fig.add_vline(
            x=step,
            line=dict(color='orange', dash='dash'),
            annotation_text="Gate Switch",
            annotation_position="top right",
        )

    # Layout adjustments
    fig.update_layout(
        title=figure_title,
        xaxis=dict(title="Episodes"),
        yaxis=dict(title="Fidelity and Rewards", side='left'),
        yaxis2=dict(
            title="Infidelity (Log Scale)",
            overlaying='y',
            side='right',
            type='log'
        ),
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
        width=1200,
        height=600,
    )

    # Save the plot as an interactive HTML file
    output_file = save_dir + "plot.html"
    fig.write_html(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    save_dir = RESULTS_DIR + "2024-02-27_19-31-17_H/"
    plot_data(save_dir, episode_length=2, figure_title="Interactive Data Visualization")



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


def plot_inferencing_data(save_dir, episode_length, figure_title='', env_data_path=''):
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
    rolling_average_window = 10
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
    
    
def generate_mock_training_data(num_steps=60000, num_switches=4):
    # Initialize lists for data
    fidelities = []
    rewards = []
    episode_ids = []
    gate_switches = []
    
    # Parameters for fidelity progression
    base_fidelity = 0.5  # Starting fidelity
    noise_std = 0.02     # Noise in fidelity
    recovery_rate = 0.0001  # How fast fidelity improves
    switch_drop = 0.3    # How much fidelity drops at switch
    
    steps_per_switch = num_steps // (num_switches + 1)
    current_episode = 0
    steps_per_episode = 10
    
    for step in range(num_steps):
        # Determine if it's a switch point
        is_switch = step > 0 and step % steps_per_switch == 0
        gate_switches.append(1 if is_switch else 0)
        
        # Calculate base fidelity progression
        progress = min(0.4, (step % steps_per_switch) * recovery_rate)
        
        if is_switch:
            base_fidelity = max(0.5, base_fidelity - switch_drop)
        
        # Add noise and ensure fidelity stays in [0,1]
        current_fidelity = min(0.99, max(0, base_fidelity + progress + np.random.normal(0, noise_std)))
        
        # Calculate reward based on fidelity improvement
        reward = current_fidelity * 2 - 1
        
        # Track episode ids
        if step % steps_per_episode == 0:
            current_episode += 1
        
        fidelities.append(current_fidelity)
        rewards.append(reward)
        episode_ids.append(current_episode)
        
        # After each switch, gradually increase base fidelity
        if is_switch:
            base_fidelity = current_fidelity
    
    # Create DataFrame
    df = pd.DataFrame({
        'Fidelity': fidelities,
        'Reward': rewards,
        'Actions': [0] * num_steps,  # Placeholder
        'Matrices': [0] * num_steps,  # Placeholder
        'Episode ID': episode_ids,
        'Gate Switch': gate_switches
    })
    
    return df
import os
if __name__ =="__main__":    
    # Generate mock data
    mock_df = generate_mock_training_data(60000, 4)

    # Save to CSV
    save_dir = "./mock_training_data/"
    os.makedirs(save_dir, exist_ok=True)
    mock_df.to_csv(save_dir + "env_data_train.csv", index=False)

    # Plot using your existing plot_data function
    plot_data(
        save_dir,
        episode_length=10,
        figure_title="Ideal Training Progress with Gate Switches",
        env_data_path="env_data_train.csv"
    )