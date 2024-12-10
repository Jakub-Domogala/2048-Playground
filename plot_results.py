import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import datetime


def plot_results(rewards, window_width):
    averages, std_devs = rolling_average_and_std(rewards, window_width)

    x_values = np.arange(window_width, len(rewards[0]) + 1)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, averages, label='Rolling Average', linewidth=2, color='blue')
    plt.fill_between(x_values, averages + std_devs, averages - std_devs, color='lightblue', alpha=0.5, label='Standard Deviation Band')
    plt.title('Q-Learning Rewards - Rolling Average and Standard Deviation')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    # plt.ylim(0, 600)
    plt.legend()
    plt.grid()
    
    safe_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"plots\Q_{safe_datetime}.png")
    plt.show()

def rolling_average_and_std(rewards, window_width):
    rewards_array = np.array(rewards)

    averages = np.convolve(rewards_array.mean(axis=0), np.ones(window_width)/window_width, mode='valid')
    std_devs = np.sqrt(np.convolve(rewards_array.var(axis=0), np.ones(window_width)/window_width, mode='valid'))

    return averages, std_devs

if __name__ == "__main__":
    rewards_file = sys.argv[1]
    window_width = int(sys.argv[2])
    rewards = np.loadtxt(rewards_file, ndmin=2)
    plot_results(rewards, window_width)