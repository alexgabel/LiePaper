import os
import json
import subprocess

# Define ranges for hyperparameters
lambda_alpha_values = [0.001, 1.0, 1000.0, 0.00001]
models = ["EncoderLieTDecoder", "EncoderLieMulTDecoder"]
transformations = [
    {"tf_range": [10, 10, 90, 1, 0, 0, 0]},  # rotation only
    {"tf_range": [10, 10, 0, 1.5, 0, 0, 0]},  # scaling only
    {"tf_range": [15, 15, 0, 1, 0, 0, 0]}  # translation only
]
output_dir = "grid_search_results"

os.makedirs(output_dir, exist_ok=True)

# Base configuration template
with open("config_template.json", "r") as f:
    base_config = json.load(f)

# Iterate over combinations of hyperparameters
for model in models:
    for tf_cfg in transformations:
        for lambda_alpha in lambda_alpha_values:
            # Update base configuration
            config = base_config.copy()
            config["arch"]["type"] = model
            config["data_loader"]["args"].update(tf_cfg)
            config["lambda_a"] = lambda_alpha
            config_name = f"{model}_tf_{tf_cfg['tf_range']}_la_{lambda_alpha}".replace(" ", "_").replace(",", "_")
            config["name"] = config_name

            # Save updated configuration
            config_path = os.path.join(output_dir, f"{config_name}.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)

            # Launch training
            subprocess.run(["python", "train.py", "-c", config_path])

print("Grid search completed!")
