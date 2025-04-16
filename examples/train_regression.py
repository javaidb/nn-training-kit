import os
import yaml
from nn_training_kit.core.trainer import train_from_config

def main():
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load base config
    config_path = os.path.join(script_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Modify config in script
    config["trainer"]["max_epochs"] = 30  # Change number of epochs
    config["trainer"]["learning_rate"] = 0.0005  # Change learning rate
    config["model"]["hidden_dims"] = [128, 64]  # Change model architecture
    config["trainer"]["hyperparameter_tuning"]["n_trials"] = 3  # Change number of trials
    
    # Train model using the modified config
    study = train_from_config(config=config)
    
    # Print best trial
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main() 