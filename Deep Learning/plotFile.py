import json
import matplotlib.pyplot as plt

def load_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def plot_training_stats(data):
    epochs = [entry['epoch'] for entry in data]
    losses = []
    accuracies = []

    for entry in data:
        avg_loss = sum(step['loss'] for step in entry['steps']) / len(entry['steps'])
        avg_accuracy = sum(step['accuracy'] for step in entry['steps']) / len(entry['steps'])
        losses.append(avg_loss)
        accuracies.append(avg_accuracy)

    # Plotting Loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plotting Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, marker='o', color='green')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()

# Example usage:
json_file_path = 'parsed_resnet_data.json'
parsed_data = load_json(json_file_path)

# Plot the training stats
plot_training_stats(parsed_data)
