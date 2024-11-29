import matplotlib.pyplot as plt
import csv

def plot_training_progress(log_file):
    epochs = []
    train_losses = []
    eval_losses = []

    # Read the training log file and extract data
    with open(log_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            epochs.append(int(row['Epoch']))
            train_losses.append(float(row['Train Loss']))
            eval_losses.append(float(row['Eval Loss']))

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue', marker='o')
    plt.plot(epochs, eval_losses, label='Eval Loss', color='red', marker='x')

    # Make sure the x-axis stays integer
    plt.xticks(epochs)

    # add the values next to the points but round them to 2 decimal places
    for i, txt in enumerate(train_losses):
        plt.annotate(round(txt, 2), (epochs[i], train_losses[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    for i, txt in enumerate(eval_losses):
        plt.annotate(round(txt, 2), (epochs[i], eval_losses[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()

# Usage:
log_file = 'training_log.csv'
plot_training_progress(log_file)
