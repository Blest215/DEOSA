import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

font = {'size': 24}
matplotlib.rc('font', **font)
exponential_moving_average_window = 25

data_datetime = "2020-10-28-21-19-28"


def read_data(agent, phase, measure):
    file_path = "data/run-{agent}_{date}_{phase}-tag-{measure}.csv".format(agent=agent,
                                                                           date=data_datetime,
                                                                           phase=phase,
                                                                           measure=measure)
    try:
        return np.array(pd.read_csv(file_path)["Value"])
    except FileNotFoundError:
        print("FileNotFound: " + file_path)
        return None


def get_data(measure, train=False):
    data = {
        "DEOSA": read_data("DEOSA", "test", measure),
        "NoReplacement": read_data("NoHandoverSelectionAgent", "test", measure),
        "Nearest": read_data("NearestSelectionAgent", "test", measure),
        "Random": read_data("RandomSelectionAgent", "test", measure)
    }
    if train:
        data["DEOSA (train)"] = read_data("DEOSA", "train", measure)
    return data


def linear_filter(value_list, window_size=3):
    def shift_sublist(values, shift):
        shifted_list = np.roll(values, shift)
        if shift > 0:
            shifted_list[:shift] = values[:shift]
        else:
            shifted_list[shift:] = values[shift:]
        return shifted_list

    filtered_list = np.copy(value_list)
    for s in range(-window_size, window_size + 1):
        if s != 0:
            filtered_list += shift_sublist(value_list, s)
    return filtered_list / (2 * window_size + 1)


def exponential_moving_average(value_list, window_size=10):
    ema_list = [value_list[0]]
    multiplier = 2 / (window_size + 1)
    for i in range(1, len(value_list)):
        ema_list.append(value_list[i] * multiplier + ema_list[-1] * (1 - multiplier))
    return ema_list


def set_axis_range(y_axis_range, x_rate=100, y_rate=0.5):
    # Axis
    plt.xticks(np.arange(0, 1000, x_rate))
    plt.yticks(np.arange(y_axis_range[0], y_axis_range[1] + 1, y_rate))
    plt.ylim(y_axis_range)


def get_agent_properties(agent):
    if "DEOSA (train)" == agent:
        color = "firebrick"
        marker = "o"
        name = "DEOSA (train)"
    elif "DEOSA (test)" == agent or "DEOSA" == agent:
        color = "orangered"
        marker = "D"
        name = "DEOSA (test)"
    elif "NoHandover" == agent or "NoReplacement" == agent:
        color = "forestgreen"
        marker = "v"
        name = "Greedy (no replacement)"
    elif "Nearest" == agent:
        color = "steelblue"
        marker = "^"
        name = "Greedy (nearest)"
    else:
        color = "gray"
        marker = ","
        name = "Random"
    return {
        "color": color,
        "marker": marker,
        "label": name
    }


def plot_reward():
    """ plot_reward: Average reward over simulations """
    x_axis = np.array(range(0, 1000))

    data = get_data("reward_mean", train=True)

    # Exponential moving average smoothing with markers
    for agent in data:
        plt.plot(x_axis, exponential_moving_average(data[agent], exponential_moving_average_window),
                 markevery=50, markersize=25, linewidth=2, **get_agent_properties(agent))

    # 0 line
    plt.axhline(0, color="black", linestyle="dotted")
    # Ticks
    set_axis_range([-0.5, 1], x_rate=100, y_rate=0.1)
    # Grid
    plt.grid(True, axis="y", color="gray", alpha=0.5, linestyle='--')

    # Label
    plt.xlabel("Simulation")
    plt.ylabel("Average Reward")

    # Legend
    plt.legend(facecolor="white", loc=4)

    # Data points
    for agent in data:
        plt.scatter(x_axis, data[agent], alpha=0.1, s=100, **get_agent_properties(agent))

    plt.show()


def plot_statistics():
    reward_mean = get_data("reward_mean")
    effectiveness_mean = get_data("effectiveness_mean")
    penalty_mean = get_data("penalty_mean")

    reward_stddev = get_data("reward_stddev")
    effectiveness_stddev = get_data("effectiveness_stddev")
    penalty_stddev = get_data("penalty_stddev")

    fig, axes = plt.subplots(2, 3, sharex='all')

    # Reward mean
    axes[0, 0].set_title("Reward")
    axes[0, 0].set_ylabel("Average")
    axes[0, 0].boxplot([reward_mean[agent] for agent in reward_mean], notch=True)
    axes[0, 0].set_xticklabels([agent for agent in reward_mean])
    axes[0, 0].yaxis.grid(True)

    # Effectiveness mean
    axes[0, 1].set_title("Effectiveness")
    axes[0, 1].boxplot([effectiveness_mean[agent] for agent in effectiveness_mean], notch=True)
    axes[0, 1].set_xticklabels([agent for agent in effectiveness_mean])
    axes[0, 1].yaxis.grid(True)

    # Penalty mean
    axes[0, 2].set_title("Penalty")
    axes[0, 2].boxplot([penalty_mean[agent] for agent in penalty_mean], notch=True)
    axes[0, 2].set_xticklabels([agent for agent in penalty_mean])
    axes[0, 2].yaxis.grid(True)

    # Reward stddev
    axes[1, 0].set_ylabel("Standard deviation")
    axes[1, 0].boxplot([reward_stddev[agent] for agent in reward_stddev], notch=True)
    axes[1, 0].set_xticklabels([agent for agent in reward_stddev], rotation=45)
    axes[1, 0].yaxis.grid(True)

    # Effectiveness stddev
    axes[1, 1].boxplot([effectiveness_stddev[agent] for agent in effectiveness_stddev], notch=True)
    axes[1, 1].set_xticklabels([agent for agent in effectiveness_stddev], rotation=45)
    axes[1, 1].yaxis.grid(True)

    # Penalty stddev
    axes[1, 2].boxplot([penalty_stddev[agent] for agent in penalty_stddev], notch=True)
    axes[1, 2].set_xticklabels([agent for agent in penalty_stddev], rotation=45)
    axes[1, 2].yaxis.grid(True)

    plt.subplots_adjust(left=0.06, bottom=0.16, right=0.99, top=0.96, wspace=0.3, hspace=0.2)

    plt.show()


def plot_execution_time():
    execution_time_mean = get_data("execution_time_mean")
    execution_time_stddev = get_data("execution_time_stddev")

    fig, axes = plt.subplots(1, 2, sharex='all')

    axes[0].boxplot([execution_time_mean[agent] for agent in execution_time_mean], notch=True)
    axes[0].set_ylabel("Average (sec)")
    axes[0].set_xticklabels([agent for agent in execution_time_mean], rotation=45)
    axes[0].yaxis.grid(True)

    axes[1].boxplot([execution_time_stddev[agent] for agent in execution_time_stddev], notch=True)
    axes[1].set_ylabel("Standard deviation")
    axes[1].set_xticklabels([agent for agent in execution_time_mean], rotation=45)
    axes[1].yaxis.grid(True)

    plt.show()


def plot_loss():
    data = {
        "max": read_data("DEOSA", "train", "loss_max")[10:],
        "min": read_data("DEOSA", "train", "loss_min")[10:],
        "mean": read_data("DEOSA", "train", "loss_mean")[10:],
        "stddev": read_data("DEOSA", "train", "loss_stddev")[10:]
    }

    # Average reward over simulations
    # x_axis = np.array(range(11, 1001))
    x_axis = np.array(range(len(data["max"])))

    fig, axes = plt.subplots(1, 4)

    axes[0].plot(x_axis, exponential_moving_average(data["max"], exponential_moving_average_window),
                 color="firebrick", linewidth=1)
    axes[0].set_ylabel("Maximum")
    axes[0].yaxis.grid(True)

    axes[1].plot(x_axis, exponential_moving_average(data["min"], exponential_moving_average_window),
                 color="firebrick", linewidth=1)
    axes[1].set_ylabel("Minimum")
    axes[1].yaxis.grid(True)

    axes[2].plot(x_axis, exponential_moving_average(data["mean"], exponential_moving_average_window),
                 color="firebrick", linewidth=1)
    axes[2].set_ylabel("Average")
    axes[2].yaxis.grid(True)

    axes[3].plot(x_axis, exponential_moving_average(data["stddev"], exponential_moving_average_window),
                 color="firebrick", linewidth=1)
    axes[3].set_ylabel("Standard deviation")
    axes[3].yaxis.grid(True)

    plt.show()


# plot_reward()
# plot_statistics()
# plot_execution_time()
# plot_loss()
