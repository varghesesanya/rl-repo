import datetime
import os
import csv
from relaqs import RESULTS_DIR
from typing import List, Dict
import json
import numpy as np
import pandas as pd
from types import MappingProxyType

l = frozenset([])
FrozenSetType = type(l)
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, MappingProxyType):
            return obj.copy()
        if isinstance(obj, FrozenSetType):
            return list(obj)
        else:
            return obj.__dict__         
        return super(NpEncoder, self).default(obj)

class SaveResults():
    def __init__(self,
                 env=None,
                 alg=None,
                 results:List[Dict]=None,
                 save_path=None,
                 save_base_path=None,
                 target_gate_string=None
                ):
        self.env = env
        self.alg = alg
        self.target_gate_string = target_gate_string
        if save_path is None:
            self.save_path = self.get_new_directory(save_base_path)
        else:
            self.save_path = save_path
    
        # Create directory if it does not exist
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        self.results = results

    def get_new_directory(self, save_base_path=None):
        if save_base_path is None:
            save_base_path = RESULTS_DIR

        path = save_base_path + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")

        if self.target_gate_string is not None:
            path = path[:-1] + "_"  + self.target_gate_string + "/"

        return path

    def save_env_transitions_train(self):
        columns = ['Fidelity', 'Rewards', 'Actions', 'Operator', 'Episode Id']
        df = pd.DataFrame(self.env.transition_history, columns=columns)  # Ensure transition_history is a valid DataFrame
        # Initialize the Gate Switch column
        gate_switch_column = [0] * len(df)  # Default all to 0

        # Mark gate switch timesteps  
        for i in range(len(df)):
            if i in self.env.gate_switch_timesteps:  # Use self.env.gate_switch_timesteps
                gate_switch_column[i] = 1

        # Add the Gate Switch column to the DataFrame
        df['Gate Switch'] = gate_switch_column

        # Save the data
        print("Data SAVED IN", self.save_path + "env_data_train.pkl")
        df.to_pickle(self.save_path + "env_data_train.pkl")
        df.to_csv(self.save_path + "env_data_train.csv", index=False)


        
    def save_env_transitions_inference(self):
        columns = ['Fidelity', 'Rewards', 'Actions', 'Operator', 'Episode Id']
        df = pd.DataFrame(self.env.transition_history, columns=columns)
        df.to_pickle(self.save_path + "env_data_inference.pkl") # easier to load than csv
        df.to_csv(self.save_path + "env_data_inference.csv", index=False) # backup in case pickle doesn't work    
    
    def save_train_results_data(self):
        with open(self.save_path+'train_results_data.json', 'w') as f:
            json.dump(self.results,f, cls=NpEncoder)

    def save_config(self, config_dict):
        config_path = self.save_path + "config.txt"
        with open(config_path, "w") as file:
            for key, value in config_dict.items():
                file.write(f"{key}: {value}\n")

    def save_model(self):
        save_model_path = self.save_path + "model_checkpoints/"
        self.alg.save(save_model_path)

    def save_results(self, train_or_inference):
        if train_or_inference == "train":
            if self.env is not None:
                self.save_env_transitions_train()
        elif train_or_inference == "inference":
            if self.env is not None:
                self.save_env_transitions_inference()       
        if self.alg is not None:
            self.save_config(self.alg.get_config().to_dict())
            self.save_model()
        if self.results is not None:
            self.save_train_results_data()
        return self.save_path
    
    def save_env_transitions_inference(self):
        columns = ['Fidelity', 'Rewards', 'Actions', 'Operator', 'Episode Id', 'Gate_Index']
        
        # Create DataFrame with gate index information
        df = pd.DataFrame(self.env.transition_history, columns=columns[:-1])  # Original columns
        
        # Save with appropriate naming that includes gate information
        base_name = "env_data_inference"
        df.to_pickle(self.save_path + f"{base_name}.pkl")
        df.to_csv(self.save_path + f"{base_name}.csv", index=False)

    def save_inference_summary(self, inference_gates):
        """Save summary of inference results for multiple gates."""
        summary_path = os.path.join(self.save_path, "inference_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("Inference Results Summary\n")
            f.write("=======================\n\n")
            
            for idx, gate in enumerate(inference_gates):
                f.write(f"Gate {idx + 1}:\n")
                transition_data = pd.DataFrame(self.env.transition_history)
                fidelities = transition_data[0]  # Assuming fidelity is first column
                
                f.write(f"Average Fidelity: {np.mean(fidelities):.4f}\n")
                f.write(f"Max Fidelity: {np.max(fidelities):.4f}\n")
                f.write(f"Min Fidelity: {np.min(fidelities):.4f}\n")
                f.write(f"Std Dev: {np.std(fidelities):.4f}\n\n")
