import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy import stats
from statannotations.Annotator import Annotator

exponential_moving_average_window = 25

data_datetime = "2022-06-13-14-51-12"


def set_font_size(font_size):
    plt.rc('font', size=int(font_size * 0.75))
    plt.rc('axes', titlesize=int(font_size * 1.5))
    plt.rc('axes', labelsize=int(font_size * 1.5))
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    plt.rc('legend', fontsize=font_size)
    plt.rc('figure', titlesize=font_size)


def read_data(agent, phase, measure):
    file_path = "data/run-{date}_{agent}_{phase}-tag-{measure}.csv".format(agent=agent,
                                                                           date=data_datetime,
                                                                           phase=phase,
                                                                           measure=measure)
    try:
        return np.array(pd.read_csv(file_path)["Value"])
    except FileNotFoundError:
        print("FileNotFound: " + file_path)
        return None


def get_test_data(measure):
    return {
        "DEOSA": read_data("DEOSA_MunchausenDQNetwork", "test", measure),
        "DoubleDQN": read_data("DEOSA_DoubleDQNetwork", "test", measure),
        "DQN": read_data("DEOSA_DQNetwork", "test", measure),
        "Greedy": read_data("Greedy", "test", measure),
        "Nearest": read_data("Nearest", "test", measure),
        "Random": read_data("Random", "test", measure)
    }


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
    plt.xticks(np.arange(0, 1001, x_rate))
    plt.yticks(np.arange(y_axis_range[0], y_axis_range[1] + 1, y_rate))
    plt.ylim(y_axis_range)


def plot_reward():
    def get_agent_properties(agent):
        if "DEOSA (train)" == agent:
            color = "indianred"
            marker = "^"
            name = "DEOSA (train)"
        elif "DEOSA (test)" == agent:
            color = "firebrick"
            marker = "*"
            name = "DEOSA (test)"
        elif "DoubleDQN (train)" == agent:
            color = "mediumblue"
            marker = "D"
            name = "DoubleDQN (train)"
        elif "DoubleDQN (test)" == agent:
            color = "dodgerblue"
            marker = "s"
            name = "DoubleDQN (test)"
        elif "DQN (train)" == agent:
            color = "limegreen"
            marker = "P"
            name = "DQN (train)"
        elif "DQN (test)" == agent:
            color = "seagreen"
            marker = "X"
            name = "DQN (test)"
        elif "Nearest" == agent:
            color = "forestgreen"
            marker = "p"
            name = "Nearest"
        elif "Greedy" == agent:
            color = "royalblue"
            marker = "."
            name = "Greedy"
        else:
            color = "gray"
            marker = ","
            name = "Random"
        return {
            "color": color,
            "marker": marker,
            "label": name
        }

    set_font_size(36)

    x_axis = np.array(range(0, 1000))

    measure = "reward_mean"
    data = {
        "Random": read_data("Random", "test", measure),
        "Nearest": read_data("Nearest", "test", measure),
        "Greedy": read_data("Greedy", "test", measure),
        "DQN (train)": read_data("DEOSA_DQNetwork", "train", measure),
        "DQN (test)": read_data("DEOSA_DQNetwork", "test", measure),
        "DoubleDQN (train)": read_data("DEOSA_DoubleDQNetwork", "train", measure),
        "DoubleDQN (test)": read_data("DEOSA_DoubleDQNetwork", "test", measure),
        "DEOSA (train)": read_data("DEOSA_MunchausenDQNetwork", "train", measure),
        "DEOSA (test)": read_data("DEOSA_MunchausenDQNetwork", "test", measure)
    }

    # Exponential moving average smoothing with markers
    for agent in data:
        plt.plot(x_axis, exponential_moving_average(data[agent], exponential_moving_average_window),
                 markevery=50, markersize=30, linewidth=2, **get_agent_properties(agent))

    # 0 line
    plt.axhline(0, color="black", linestyle="-")
    # Ticks
    set_axis_range([-0.5, 0.8], x_rate=100, y_rate=0.1)
    # Grid
    plt.grid(True, axis="both", color="gray", alpha=0.5, linestyle='--')

    # Label
    plt.xlabel("Simulation")
    plt.ylabel("Average Reward")

    # Legend
    plt.legend(facecolor="white", loc=4)

    # Data points
    for agent in data:
        plt.scatter(x_axis, data[agent], alpha=0.1, s=200, **get_agent_properties(agent))

    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.99, top=0.98, wspace=0.3, hspace=0.2)

    plt.show()


def plot_statistics():
    set_font_size(24)

    reward_mean = get_test_data("reward_mean")
    effectiveness_mean = get_test_data("effectiveness_mean")
    penalty_mean = get_test_data("penalty_mean")

    reward_stddev = get_test_data("reward_stddev")
    effectiveness_stddev = get_test_data("effectiveness_stddev")
    penalty_stddev = get_test_data("penalty_stddev")

    algorithms = ['Random', 'Nearest', 'Greedy', 'DQN', 'DoubleDQN', 'DEOSA']

    data = []
    for i in range(1000):
        for algorithm in algorithms:
            data.append({
                'Algorithm': algorithm,
                'Reward_mean': reward_mean[algorithm][i], 'Reward_stddev': reward_stddev[algorithm][i],
                'Effectiveness_mean': effectiveness_mean[algorithm][i],
                'Effectiveness_stddev': effectiveness_stddev[algorithm][i],
                'Penalty_mean': penalty_mean[algorithm][i], 'Penalty_stddev': penalty_stddev[algorithm][i]
            })
    df = pd.DataFrame(data)

    fig, axes = plt.subplots(3, 2, sharex='all')

    def plot_stat(row, value, alternative):
        print(value)

        mean = '%s_mean' % value
        stddev = '%s_stddev' % value

        plot = sns.boxplot(data=df, x='Algorithm', y=mean, order=algorithms, ax=axes[row, 0], notch=True)
        sns.boxplot(data=df, x='Algorithm', y=stddev, order=algorithms, ax=axes[row, 1], notch=True)

        print(stats.normaltest(df[df['Algorithm'] == 'DEOSA'][mean]))
        print(stats.normaltest(df[df['Algorithm'] == 'DoubleDQN'][mean]))
        print(stats.normaltest(df[df['Algorithm'] == 'Greedy'][mean]))

        annotator = Annotator(plot,
                              pairs=[
                                  ('DEOSA', 'DQN'),
                                  ('DEOSA', 'DoubleDQN'),
                              ],
                              data=df, x='Algorithm', y=mean, order=algorithms)
        test = 'Mann-Whitney-gt' if alternative == 'greater' else 'Mann-Whitney-ls'
        annotator.configure(test=test, text_format='simple', show_test_name=False)
        annotator.apply_test(alternative=alternative)
        annotator.annotate()

    plot_stat(0, 'Reward', 'less')
    plot_stat(1, 'Effectiveness', 'less')
    plot_stat(2, 'Penalty', 'greater')

    for i in range(3):
        for j in range(2):
            axes[i][j].set_xlabel(None)
            axes[i][j].set_ylabel(None)
            axes[i][j].yaxis.grid(True)

    # Reward mean
    axes[0, 0].set_title("Average")
    axes[0, 0].set_ylabel("Reward")

    # Reward stddev
    axes[0, 1].set_title("Standard deviation")

    # Effectiveness mean
    axes[1, 0].set_ylabel("Effectiveness")

    # Effectiveness stddev

    # Penalty mean
    axes[2, 0].set_ylabel("Penalty")
    axes[2, 0].set_xticklabels(algorithms, rotation=45)

    # Penalty stddev
    axes[2, 1].set_xticklabels(algorithms, rotation=45)

    fig.align_ylabels()
    fig.align_xlabels()

    plt.subplots_adjust(left=0.1, bottom=0.11, right=0.99, top=0.97, wspace=0.3, hspace=0.2)

    plt.show()


def plot_execution_time():
    set_font_size(36)

    execution_time_mean = get_test_data("execution_time_mean")
    execution_time_stddev = get_test_data("execution_time_stddev")

    algorithms = ['Random', 'Nearest', 'Greedy', 'DQN', 'DoubleDQN', 'DEOSA']

    data = []
    for i in range(1000):
        for algorithm in algorithms:
            data.append({
                'Algorithm': algorithm,
                'Execution_time_mean': execution_time_mean[algorithm][i],
                'Execution_time_stddev': execution_time_stddev[algorithm][i]
            })
    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, sharex='all')

    sns.boxplot(data=df, x='Algorithm', y='Execution_time_mean', order=algorithms,
                ax=axes[0], notch=True, showfliers=False)
    sns.boxplot(data=df, x='Algorithm', y='Execution_time_stddev', order=algorithms, ax=axes[1],
                notch=True, showfliers=False)

    for i in range(2):
        axes[i].set_xlabel(None)
        axes[i].set_ylabel(None)
        axes[i].yaxis.grid(True)

    axes[0].set_title("Average (sec)")
    axes[0].set_ylabel("Execution time")
    axes[0].set_xticklabels(algorithms, rotation=45)

    axes[1].set_title("Standard deviation")
    axes[1].set_xticklabels(algorithms, rotation=45)

    plt.subplots_adjust(left=0.08, bottom=0.15, right=0.99, top=0.95, wspace=0.4, hspace=0.2)

    plt.show()


# plot_reward()
plot_statistics()
# plot_execution_time()
