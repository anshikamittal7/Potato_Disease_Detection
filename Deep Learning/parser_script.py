import re
import json

def parse_training_logs(file_path):
    pattern = re.compile(r'Epoch (\d+)/\d+\s+\n(?:\s+\d+/\d+\s+\[.*?\] - ETA: .*? - loss: ([-+]?\d*\.\d+|\d+) - accuracy: ([-+]?\d*\.\d+|\d+)\n)+')

    logs = []

    with open(file_path, 'r') as file:
        file_content = file.read()
        matches = pattern.finditer(file_content)

        for match in matches:
            epoch_data = {
                'epoch': int(match.group(1)),
                'steps': []
            }

            step_pattern = re.compile(r'\s+(\d+)/\d+\s+\[.*?\] - ETA: .*? - loss: ([-+]?\d*\.\d+|\d+) - accuracy: ([-+]?\d*\.\d+|\d+)')

            step_matches = step_pattern.finditer(match.group(0))
            for step_match in step_matches:
                step_data = {
                    'step': int(step_match.group(1)),
                    'loss': float(step_match.group(2)),
                    'accuracy': float(step_match.group(3))
                }
                epoch_data['steps'].append(step_data)

            logs.append(epoch_data)

    return logs

def save_to_json(data, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=2)

# Example usage:
file_path = 'sequential_status_train.txt'
output_json_file = 'parsed_sequential_data.json'

parsed_data = parse_training_logs(file_path)

# Save the parsed data to a JSON file
save_to_json(parsed_data, output_json_file)

# parsed_data = parse_training_logs(file_path)

# # Access the parsed data
# for epoch_data in parsed_data:
#     print(f"Epoch: {epoch_data['epoch']}")
#     for step_data in epoch_data['steps']:
#         print(f"  Step {step_data['step']}: Loss = {step_data['loss']}, Accuracy = {step_data['accuracy']}")
