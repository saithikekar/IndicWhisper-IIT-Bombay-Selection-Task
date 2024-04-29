import subprocess
import json


model_path = "/mnt/e/vistaar/hindi_models/whisper-large-hi-noldcil"
manifest_path = "/mnt/e/vistaar/manifest.json"
gpu_index = -1  # Using CPU
batch_size = 1  # Batch Size 1
language_code = "hi"  # Hindi = hi

# Runnning Evaluation Script 
command = [
    "python", "evaluation.py",
    "--model_path", model_path,
    "--manifest_path", manifest_path,
    "--manifest_name", "Kathbath",
    "--device", str(gpu_index),
    "--batch_size", str(batch_size),
    "--language", language_code
]

# Execute the command
output = subprocess.run(command, capture_output=True, text=True)

# Print the entire output
print("Output from evaluation script:")
print(output.stdout)

# Extract WER from output
wer_value = None
output_lines = output.stdout.split("\n")
for line in output_lines:
    if line.startswith("WER:"):
        wer_value = float(line.split(":")[1].strip())
        break

if wer_value is not None:
    print("Word Error Rate (WER):", wer_value)
else:
    print("WER value not found in output.")
