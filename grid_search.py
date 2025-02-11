import json
import os

# Hyperparameter grid
latent_dims = [25, 81]
channels = [1, 4]
tf_ranges = [
    [0, 0, 20, 1, 0, 0, 0],  # Small rotation
    [0, 0, 0, 1.2, 0, 0, 0],  # Scaling
    [0, 0, 90, 1, 0, 0, 0]   # Large rotation
]
lambdas_a = [0.001, 1.0, 1000.0]
lambdas_lasso = [0.001, 0.1]
lambdas_z = [0.001]

# Output directories
config_dir = "configs"
bash_script_path = "run_grid_search.sh"

os.makedirs(config_dir, exist_ok=True)

# Generate configurations
experiment_count = 0
with open(bash_script_path, "w") as bash_script:
    bash_script.write("#!/bin/bash\n\n")
    for latent_dim in latent_dims:
        for channel in channels:
            for tf_range in tf_ranges:
                for lambda_a in lambdas_a:
                    for lambda_lasso in lambdas_lasso:
                        for lambda_z in lambdas_z:
                            experiment_name = f"latent{latent_dim}_chan{channel}_rot{tf_range[2]}_a{lambda_a}_lasso{lambda_lasso}_z{lambda_z}"
                            config = {
                                "name": experiment_name,
                                "n_gpu": 1,
                                "arch": {
                                    "type": "EncoderLieTDecoder",
                                    "args": {
                                        "hidden_sizes": [512, 256, 128, 64],
                                        "t_hidden_sizes": [512, 256, 128, 64],
                                        "latent_dim": latent_dim,
                                        "channels": channel,
                                        "non_affine": False,
                                        "dropout": False
                                    }
                                },
                                "data_loader": {
                                    "type": "SuperSyMNISTDataLoader",
                                    "args": {
                                        "data_dir": "data/",
                                        "batch_size": 512,
                                        "shuffle": True,
                                        "validation_split": 0.2,
                                        "num_workers": 2,
                                        "tf_range": tf_range
                                    }
                                },
                                "optimizer": {
                                    "type": "Adam",
                                    "args": {
                                        "lr": 0.001,
                                        "lr_a": 0.01,
                                        "weight_decay": 0,
                                        "amsgrad": True
                                    }
                                },
                                "loss": "loss_all",
                                "lambda_recon": 1.0,
                                "lambda_z": lambda_z,
                                "lambda_lasso": lambda_lasso,
                                "lambda_a": lambda_a,
                                "metrics": [
                                    "mse_loss",
                                    "generator_mse",
                                    "drift_mse",
                                    "diffusion_mse",
                                    "loss_z",
                                    "loss_alpha",
                                    "loss_recon",
                                    "loss_lasso"
                                ],
                                "lr_scheduler": {
                                    "type": "StepLR",
                                    "args": {
                                        "step_size": 100,
                                        "gamma": 0.1
                                    }
                                },
                                "trainer": {
                                    "epochs": 100,
                                    "save_dir": f"saved/{experiment_name}/",
                                    "save_period": 10,
                                    "verbosity": 2,
                                    "monitor": "min val_loss",
                                    "early_stop": 100,
                                    "tensorboard": True
                                }
                            }
                            # Save the config file
                            config_path = os.path.join(config_dir, f"{experiment_name}.json")
                            with open(config_path, "w") as config_file:
                                json.dump(config, config_file, indent=4)

                            # Add the command to the bash script
                            bash_script.write(f"python train.py -c {config_path}\n")
                            experiment_count += 1

print(f"Generated {experiment_count} experiments.")
print(f"Configurations are saved in '{config_dir}/'.")
print(f"Bash script to run experiments: {bash_script_path}")