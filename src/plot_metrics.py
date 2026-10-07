
import json
import matplotlib.pyplot as plt
import os

def plot_metrics(json_path="results/checkpoint-3200/trainer_state.json", output_file="training_metrics.png"):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    history = data['log_history']
    
    steps = []
    loss = []
    lr = []
    
    eval_steps = []
    eval_loss = []
    eval_per = []

    for entry in history:
        if 'loss' in entry:
            steps.append(entry['step'])
            loss.append(entry['loss'])
            if 'learning_rate' in entry:
                lr.append(entry['learning_rate'])
        
        if 'eval_loss' in entry:
            eval_steps.append(entry['step'])
            eval_loss.append(entry['eval_loss'])
        
        if 'eval_per' in entry:
            eval_per.append(entry['eval_per'])

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(steps, loss, color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    if eval_steps:
        ax1.plot(eval_steps, eval_loss, color='tab:orange', linestyle='--', label='Eval Loss')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Learning Rate', color=color)  # we already handled the x-label with ax1
    ax2.plot(steps, lr, color=color, linestyle=':', label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Training Metrics')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    print(f"Saving plot to {output_file}...")
    plt.savefig(output_file)
    print("Done.")

if __name__ == "__main__":
    plot_metrics()
