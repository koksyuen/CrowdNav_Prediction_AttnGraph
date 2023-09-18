from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import seaborn as sns
import glob
import os
import scipy.ndimage
from scipy.signal import butter, filtfilt
from scipy.signal import savgol_filter

import scienceplots

# plt.style.use(['science', 'ieee', 'nature'])
plt.style.use(['science'])


def boxplot_2d(x, y, ax, whis=1.5):
    xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
    ylimits = [np.percentile(y, q) for q in (25, 50, 75)]

    ##the box
    box = Rectangle(
        (xlimits[0], ylimits[0]),
        (xlimits[2] - xlimits[0]),
        (ylimits[2] - ylimits[0]),
        ec='k',
        zorder=0
    )
    ax.add_patch(box)

    ##the x median
    vline = Line2D(
        [xlimits[1], xlimits[1]], [ylimits[0], ylimits[2]],
        color='k',
        zorder=1
    )
    ax.add_line(vline)

    ##the y median
    hline = Line2D(
        [xlimits[0], xlimits[2]], [ylimits[1], ylimits[1]],
        color='k',
        zorder=1
    )
    ax.add_line(hline)

    ##the central point
    ax.plot([xlimits[1]], [ylimits[1]], color='k', marker='o')

    ##the x-whisker
    ##defined as in matplotlib boxplot:
    ##As a float, determines the reach of the whiskers to the beyond the
    ##first and third quartiles. In other words, where IQR is the
    ##interquartile range (Q3-Q1), the upper whisker will extend to
    ##last datum less than Q3 + whis*IQR). Similarly, the lower whisker
    ####will extend to the first datum greater than Q1 - whis*IQR. Beyond
    ##the whiskers, data are considered outliers and are plotted as
    ##individual points. Set this to an unreasonably high value to force
    ##the whiskers to show the min and max values. Alternatively, set this
    ##to an ascending sequence of percentile (e.g., [5, 95]) to set the
    ##whiskers at specific percentiles of the data. Finally, whis can
    ##be the string 'range' to force the whiskers to the min and max of
    ##the data.
    iqr = xlimits[2] - xlimits[0]

    ##left
    left = np.min(x[x > xlimits[0] - whis * iqr])
    whisker_line = Line2D(
        [left, xlimits[0]], [ylimits[1], ylimits[1]],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [left, left], [ylimits[0], ylimits[2]],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_bar)

    ##right
    right = np.max(x[x < xlimits[2] + whis * iqr])
    whisker_line = Line2D(
        [right, xlimits[2]], [ylimits[1], ylimits[1]],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [right, right], [ylimits[0], ylimits[2]],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_bar)

    ##the y-whisker
    iqr = ylimits[2] - ylimits[0]

    ##bottom
    bottom = np.min(y[y > ylimits[0] - whis * iqr])
    whisker_line = Line2D(
        [xlimits[1], xlimits[1]], [bottom, ylimits[0]],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0], xlimits[2]], [bottom, bottom],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_bar)

    ##top
    top = np.max(y[y < ylimits[2] + whis * iqr])
    whisker_line = Line2D(
        [xlimits[1], xlimits[1]], [top, ylimits[2]],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0], xlimits[2]], [top, top],
        color='k',
        zorder=1
    )
    ax.add_line(whisker_bar)

    ##outliers
    mask = (x < left) | (x > right) | (y < bottom) | (y > top)
    ax.scatter(
        x[mask], y[mask],
        facecolors='none', edgecolors='k'
    )


def apf_3d():
    # Generate image data
    image = np.load('./apf/90.npy')
    print('max: {}, min: {}'.format(np.max(image), np.min(image)))

    # Define shifted coordinate ranges
    x_range = (-5, 5)
    y_range = (-5, 5)
    z_range = (-0.1, 1)

    # Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=28)

    # Surface plot
    x, y = np.meshgrid(np.linspace(x_range[0], x_range[1], 100),
                       np.linspace(y_range[0], y_range[1], 100))
    z = x * np.exp(-x ** 2 - y ** 2)

    surf = ax.plot_surface(x, y, image, cmap='coolwarm', linewidth=0, antialiased=False)

    # Contour plots
    cset = ax.contourf(x, y, image, zdir='z', offset=-0.1, cmap='gray', levels=500)

    # Axis limits
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(z_range)

    # Axis labels and plot title
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('Potential')
    # ax.set_title('Surface & Contour Plot')

    # Add colorbar
    fig.colorbar(surf, shrink=0.6, aspect=30, pad=0.1)

    plt.show()


def fail_rate():
    h10 = np.load('./result/Success_H10_EMO.npy')
    h10_emo = np.load('./result/Success_H10_EMO.npy')
    h15 = np.load('./result/Success_H15.npy')
    h15_emo = np.load('./result/Success_H15_EMO.npy')
    h20 = np.load('./result/Success_H20.npy')
    h20_emo = np.load('./result/Success_H20_EMO.npy')

    def cal_fail_rate(data):
        num_non_zero = np.count_nonzero(data)
        num_false = len(data) - num_non_zero
        return num_false / len(data)

    fail_rate_h10 = cal_fail_rate(h10)
    fail_rate_h10_emo = cal_fail_rate(h10_emo)
    fail_rate_h15 = cal_fail_rate(h15)
    fail_rate_h15_emo = cal_fail_rate(h15_emo)
    fail_rate_h20 = cal_fail_rate(h20)
    fail_rate_h20_emo = cal_fail_rate(h20_emo)

    # Sample data
    group_labels = ['10', '15', '20']
    values_1 = [fail_rate_h10, fail_rate_h15, fail_rate_h20]
    values_2 = [fail_rate_h10_emo, fail_rate_h15_emo, fail_rate_h20_emo]

    # Set up positions for the bars
    x = np.arange(len(group_labels)) / 3
    width = 0.1  # Width of the bars

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Plot the bars for each group
    bars1 = ax.bar(x - width / 2, values_1, width, label='emotionless', color='#ff8383', alpha=0.7)
    bars2 = ax.bar(x + width / 2, values_2, width, label='emotion', color='#6592cd', alpha=0.7)

    ax.set_ylim(0, 0.43)
    # Adding labels, title, and legend
    ax.set_xlabel('Number of Humans')
    ax.set_ylabel('Failure Rate')
    # ax.set_title('Grouped Bar Plot')
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.legend(loc='upper left')

    # Adding data labels on top of the bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_labels(bars1)
    add_labels(bars2)

    # Set grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()

    # Display the plot
    plt.show()


def sgan_result_box():
    human10 = np.load('./sgan_result/10humans.npz')
    human15 = np.load('./sgan_result/15humans.npz')
    human20 = np.load('./sgan_result/20humans.npz')

    ADE_10 = human10['ade']
    ADE_15 = human15['ade']
    ADE_20 = human20['ade']

    FDE_10 = human10['fde']
    FDE_15 = human15['fde']
    FDE_20 = human20['fde']

    error_type = np.concatenate((np.repeat('ade', len(ADE_10) + len(ADE_15) + len(ADE_20)),
                                 np.repeat('fde', len(FDE_10) + len(FDE_15) + len(FDE_20))))

    human_num = np.concatenate((np.repeat(10, len(ADE_10)), np.repeat(15, len(ADE_15)),
                                np.repeat(20, len(ADE_20)), np.repeat(10, len(FDE_10)),
                                np.repeat(15, len(FDE_15)), np.repeat(20, len(FDE_20))))

    error = np.concatenate((ADE_10, ADE_15, ADE_20, FDE_10, FDE_15, FDE_20))

    # Combine the arrays and labels into a dataframe
    data = pd.DataFrame({'Displacement Error': error, 'Error': error_type, 'Number of Humans': human_num})

    sns.boxplot(x=data['Number of Humans'],
                y=data['Displacement Error'],
                hue=data['Error'])

    plt.show()


def sgan_result_violin():
    human10 = np.load('./sgan_result/10humans.npz')
    human15 = np.load('./sgan_result/15humans.npz')
    human20 = np.load('./sgan_result/20humans.npz')

    ADE_10 = human10['ade']
    ADE_15 = human15['ade']
    ADE_20 = human20['ade']

    FDE_10 = human10['fde']
    FDE_15 = human15['fde']
    FDE_20 = human20['fde']

    error_type = np.concatenate((np.repeat('ade', len(ADE_10) + len(ADE_15) + len(ADE_20)),
                                 np.repeat('fde', len(FDE_10) + len(FDE_15) + len(FDE_20))))

    human_num = np.concatenate((np.repeat(10, len(ADE_10)), np.repeat(15, len(ADE_15)),
                                np.repeat(20, len(ADE_20)), np.repeat(10, len(FDE_10)),
                                np.repeat(15, len(FDE_15)), np.repeat(20, len(FDE_20))))

    error = np.concatenate((ADE_10, ADE_15, ADE_20, FDE_10, FDE_15, FDE_20))

    # Combine the arrays and labels into a dataframe
    data = pd.DataFrame({'Displacement Error': error, 'Error': error_type, 'Number of Humans': human_num})

    sns.violinplot(x=data['Number of Humans'],
                   y=data['Displacement Error'],
                   hue=data['Error'])

    plt.show()


def sgan_result2():
    human10 = np.load('./sgan_result/10humans.npz')
    human15 = np.load('./sgan_result/15humans.npz')
    human20 = np.load('./sgan_result/20humans.npz')

    ADE_10 = human10['ade']
    ADE_15 = human15['ade']
    ADE_20 = human20['ade']

    FDE_10 = human10['fde']
    FDE_15 = human15['fde']
    FDE_20 = human20['fde']

    # ade
    means_ade = [np.mean(ADE_10), np.mean(ADE_15), np.mean(ADE_20)]
    std_ade = [np.std(ADE_10), np.std(ADE_15), np.std(ADE_20)]

    # fde
    means_fde = [np.mean(FDE_10), np.mean(FDE_15), np.mean(FDE_20)]
    std_fde = [np.std(FDE_10), np.std(FDE_15), np.std(FDE_20)]

    labels = [10, 15, 20]

    # Set the bar width
    bar_width = 0.1

    # Set the positions of the bars on the x-axis
    x_pos_group1 = np.linspace(0, 0.6, len(means_ade))
    x_pos_group2 = x_pos_group1 + bar_width
    x_pos_group_avg = (x_pos_group1 + x_pos_group2) / 2

    # Set the color for the bars
    bar_color = ['#ff8383', '#6592cd']

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(5, 5))

    bars_group1 = ax.bar(x_pos_group1, means_ade, bar_width,
                         yerr=std_ade, align='center',
                         alpha=0.7, label='ADE',
                         color=bar_color[0], edgecolor=bar_color[0])

    bars_group2 = ax.bar(x_pos_group2, means_fde, bar_width,
                         yerr=std_fde, align='center',
                         alpha=0.7, label='FDE',
                         color=bar_color[1], edgecolor=bar_color[1])

    # Add error bars
    ax.errorbar(x_pos_group1, means_ade, yerr=std_ade,
                fmt='none', elinewidth=1.5, capsize=5, color=bar_color[0])

    ax.errorbar(x_pos_group2, means_fde, yerr=std_fde,
                fmt='none', elinewidth=1.5, capsize=5, color=bar_color[1])

    # Add labels, title, and axes ticks
    ax.set_ylabel('Displacement Error (m)', fontsize=12)
    ax.set_xlabel('Number of Humans', fontsize=12)
    ax.set_xticks(x_pos_group_avg)
    ax.set_xticklabels(labels, fontsize=11)
    ax.tick_params(axis='x', length=0)
    ax.legend(fontsize=10)

    # Set grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.show()


def discomfort_discount_scatter():
    emotionless = np.load('./result/H20.npy')
    emotion = np.load('./result/H20_EMO.npy')

    # Create scatter plot
    plt.scatter(emotionless[:, 0], emotionless[:, 1], c='red', label='Emotionless', alpha=0.2)
    plt.scatter(emotion[:, 0], emotion[:, 1], c='blue', label='Emotion', alpha=0.2)

    # Calculate kernel density estimation (KDE) using scipy.stats.gaussian_kde
    kde_emotionless = gaussian_kde(emotionless.T)
    kde_emotion = gaussian_kde(emotion.T)

    # Create a grid of points for contour plotting
    x, y = np.meshgrid(np.linspace(min(emotionless[:, 0]), max(emotionless[:, 0]), 100),
                       np.linspace(min(emotionless[:, 1]), max(emotionless[:, 1]), 100))

    # Evaluate KDE on the grid of points
    z_emotionless = kde_emotionless(np.vstack([x.ravel(), y.ravel()]))
    z_emotionless = z_emotionless.reshape(x.shape)

    z_emotion = kde_emotion(np.vstack([x.ravel(), y.ravel()]))
    z_emotion = z_emotion.reshape(x.shape)

    # Add contours using the evaluated KDE values
    plt.contour(x, y, z_emotionless, colors='red', alpha=0.2)
    plt.contour(x, y, z_emotion, colors='blue', alpha=0.2)

    # Add labels and legend
    plt.xlabel('time (s)')
    plt.ylabel('discomfort penalty')
    plt.title('Scatter Plot of Two Arrays')
    plt.legend()

    # Show the plot
    plt.show()


def discomfort_discount_box():
    emotionless = np.load('./result/H20.npy')
    emotion = np.load('./result/H20_EMO.npy')

    fig, ax1 = plt.subplots(figsize=(5, 5))
    boxplot_2d(emotion[0], emotion[1], ax=ax1, whis=1)

    plt.show()


def discomfort_distance_seaborn():
    emotionless = 'emotionless'
    emotion = 'emotion'

    emotionless_H20 = np.load('./result/H20.npy')
    emotion_H20 = np.load('./result/H20_EMO.npy')
    emotionless_H15 = np.load('./result/H15.npy')
    emotion_H15 = np.load('./result/H15_EMO.npy')
    emotionless_H10 = np.load('./result/H10.npy')
    emotion_H10 = np.load('./result/H10_EMO.npy')

    time = np.concatenate((emotionless_H20[:, 0], emotion_H20[:, 0],
                           emotionless_H15[:, 0], emotion_H15[:, 0],
                           emotionless_H10[:, 0], emotion_H10[:, 0]))
    discomfort_penalty = np.concatenate((emotionless_H20[:, 1], emotion_H20[:, 1],
                                         emotionless_H15[:, 1], emotion_H15[:, 1],
                                         emotionless_H10[:, 1], emotion_H10[:, 1]))

    emotion_detect = np.concatenate((np.repeat(emotionless, emotionless_H20.shape[0]),
                                     np.repeat(emotion, emotion_H20.shape[0]),
                                     np.repeat(emotionless, emotionless_H15.shape[0]),
                                     np.repeat(emotion, emotion_H15.shape[0]),
                                     np.repeat(emotionless, emotionless_H10.shape[0]),
                                     np.repeat(emotion, emotion_H10.shape[0])
                                     ))

    human_num = np.concatenate((np.repeat(20, emotionless_H20.shape[0]),
                                np.repeat(20, emotion_H20.shape[0]),
                                np.repeat(15, emotionless_H15.shape[0]),
                                np.repeat(15, emotion_H15.shape[0]),
                                np.repeat(10, emotionless_H10.shape[0]),
                                np.repeat(10, emotion_H10.shape[0])
                                ))

    # Combine the arrays and labels into a dataframe
    data = pd.DataFrame({'Timespan (s)': time, 'Discomfort Penalty': discomfort_penalty,
                         'Emotion': emotion_detect, 'Number of Humans': human_num})

    # sns.boxplot(x=data['Number of Humans'],
    #             y=data['Discomfort Penalty'],
    #             hue=data['Emotion'])

    sns.boxplot(x=data['Number of Humans'],
                y=data['Timespan (s)'],
                hue=data['Emotion'],
                palette=['#ff8383', '#6592cd'],
                width=0.4,
                dodge=True)

    # Modify legend position
    plt.legend(loc='upper right')
    plt.xlim([-0.5, 3.0])

    # Remove legend title
    plt.legend(title='')

    plt.show()


def discomfort_distance_seaborn_violin():
    emotionless_H20 = np.load('./result/H20.npy')
    emotion_H20 = np.load('./result/H20_EMO.npy')
    emotionless_H15 = np.load('./result/H15.npy')
    emotion_H15 = np.load('./result/H15_EMO.npy')
    emotionless_H10 = np.load('./result/H10.npy')
    emotion_H10 = np.load('./result/H10_EMO.npy')

    time = np.concatenate((emotionless_H20[:, 0], emotion_H20[:, 0],
                           emotionless_H15[:, 0], emotion_H15[:, 0],
                           emotionless_H10[:, 0], emotion_H10[:, 0]))
    discomfort_penalty = np.concatenate((emotionless_H20[:, 1], emotion_H20[:, 1],
                                         emotionless_H15[:, 1], emotion_H15[:, 1],
                                         emotionless_H10[:, 1], emotion_H10[:, 1]))

    emotion_detect = np.concatenate((np.repeat('emotionless', emotionless_H20.shape[0]),
                                     np.repeat('emotion', emotion_H20.shape[0]),
                                     np.repeat('emotionless', emotionless_H15.shape[0]),
                                     np.repeat('emotion', emotion_H15.shape[0]),
                                     np.repeat('emotionless', emotionless_H10.shape[0]),
                                     np.repeat('emotion', emotion_H10.shape[0])
                                     ))

    human_num = np.concatenate((np.repeat(20, emotionless_H20.shape[0]),
                                np.repeat(20, emotion_H20.shape[0]),
                                np.repeat(15, emotionless_H15.shape[0]),
                                np.repeat(15, emotion_H15.shape[0]),
                                np.repeat(10, emotionless_H10.shape[0]),
                                np.repeat(10, emotion_H10.shape[0])
                                ))

    # Combine the arrays and labels into a dataframe
    data = pd.DataFrame({'Time (s)': time, 'Discomfort Penalty': discomfort_penalty,
                         'Emotion': emotion_detect, 'Number of Humans': human_num})

    sns.violinplot(x=data['Number of Humans'],
                   y=data['Discomfort Penalty'],
                   hue=data['Emotion'],
                   palette=['#ff8383', '#6592cd'],
                   width=0.4,
                   dodge=True)

    # sns.violinplot(x=data['Number of Humans'],
    #                y=data['Time (s)'],
    #                hue=data['Emotion'])

    # Modify legend position
    plt.legend(loc='lower left')

    # Remove legend title
    plt.legend(title='')

    plt.show()


def mean_reward():
    # Load and concatenate the CSV files
    files = glob.glob('./training_result/reward/*.csv')

    df_list = []
    for file_name in files:
        df = pd.read_csv(file_name)
        filename = os.path.basename(file_name)
        df['file'] = os.path.splitext(filename)[0]  # Add a new column 'file' with the filename (without extension)
        df_list.append(df)

    combined_df = pd.concat(df_list)

    # Apply smoothing to the line using rolling mean
    smoothed_df = combined_df.groupby('file').rolling(window=10, center=False, min_periods=1).mean().reset_index()

    line_styles = {'H0': '', 'H05': '',
                   'H10': '', 'H10(EMO)': (3, 2),
                   'H15': '', 'H15(EMO)': (3, 2),
                   'H20': '', 'H20(EMO)': (3, 2)}

    line_colors = {'H0': '#656565', 'H05': '#9a8c75',
                   'H10': '#6592cd', 'H10(EMO)': '#6592cd',
                   'H15': '#ff8383', 'H15(EMO)': '#ff8383',
                   'H20': '#b1e899', 'H20(EMO)': '#b1e899'}

    # Generate line plot using Seaborn
    sns.lineplot(data=smoothed_df, x='Step', y='Value', hue='file', style='file', dashes=line_styles,
                 palette=line_colors)
    sns.lineplot(data=combined_df, x='Step', y='Value', hue='file', alpha=0.2, legend=False, palette=line_colors)

    plt.xlabel('Timestep')
    plt.ylabel('Mean Episode Reward')
    plt.xlim(int(-3e5), int(9e6))

    # Modify legend position
    plt.legend(loc='lower right')

    # Remove legend title
    plt.legend(title='')
    plt.show()


def mean_timestep():
    # Load and concatenate the CSV files
    files = glob.glob('./training_result/length/*.csv')

    df_list = []
    for file_name in files:
        df = pd.read_csv(file_name)
        filename = os.path.basename(file_name)
        df['file'] = os.path.splitext(filename)[0]  # Add a new column 'file' with the filename (without extension)
        df_list.append(df)

    combined_df = pd.concat(df_list)

    # Apply smoothing to the line using rolling mean
    smoothed_df = combined_df.groupby('file').rolling(window=10, center=False, min_periods=1).mean().reset_index()

    line_styles = {'H0': '', 'H05': '',
                   'H10': '', 'H10(EMO)': (3, 2),
                   'H15': '', 'H15(EMO)': (3, 2),
                   'H20': '', 'H20(EMO)': (3, 2)}

    line_colors = {'H0': 'black', 'H05': 'brown',
                   'H10': '#6592cd', 'H10(EMO)': '#6592cd',
                   'H15': '#ff8383', 'H15(EMO)': '#ff8383',
                   'H20': '#b1e899', 'H20(EMO)': '#b1e899'}

    # Generate line plot using Seaborn
    sns.lineplot(data=smoothed_df, x='Step', y='Value', hue='file', style='file', dashes=line_styles,
                 palette=line_colors)
    sns.lineplot(data=combined_df, x='Step', y='Value', hue='file', alpha=0.2, legend=False, palette=line_colors)

    plt.xlabel('Timestep')
    plt.ylabel('Mean Episode Length (Timestep)')

    # Modify legend position
    plt.legend(loc='lower right')

    # Remove legend title
    plt.legend(title='')
    plt.show()


def test_case_viz():
    # H0: (time, comfort distance, distance)
    emotion = np.load('./test_case_result/15_20000068/H15_EMO.npy')
    emotionless = np.load('./test_case_result/15_20000068/H15.npy')

    time_limit = 50
    emotion = emotion[:time_limit, 0]
    emotionless = emotionless[:time_limit, 0]

    comfort_distance = pd.DataFrame({'Time (s)': emotion[:, 0], 'Distance (m)': emotion[:, 1],
                                     'Type': 'comfort distance'})
    emotion_distance = pd.DataFrame({'Time (s)': emotion[:, 0], 'Distance (m)': emotion[:, 2],
                                     'Type': 'emotion'})
    emotionless_distance = pd.DataFrame({'Time (s)': emotionless[:, 0], 'Distance (m)': emotionless[:, 2],
                                         'Type': 'emotionless'})

    combined_df = pd.concat([comfort_distance, emotion_distance, emotionless_distance])

    line_colors = {'comfort distance': 'black',
                   'emotion': 'blue',
                   'emotionless': 'red'}

    sns.lineplot(data=combined_df, x='Time (s)', y='Distance (m)', hue='Type', palette=line_colors)

    # plt.xlabel('Timestep')
    # plt.ylabel('Mean Episode Reward')
    # plt.xlim(int(-3e5), int(9e6))

    # Modify legend position
    # plt.legend(loc='lower right')

    # Remove legend title
    plt.legend(title='')
    plt.show()


if __name__ == '__main__':
    test_case_viz()
