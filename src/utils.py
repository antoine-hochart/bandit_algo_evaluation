import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns


# Data ################################################################

def read_data():
    """ Return a dataframe with 2 columns: movieId (actions) and rating (reward) """
    data = pd.read_csv('data/ratings.csv', header=0,
                       usecols=['movieId', 'rating'])
    return data


def preprocess_data(data, num_ratings, num_movies):
    """ Make each movieId/action uniformly distributed """
    # filters out movies with less than `num_ratings` ratings
    movies = data.groupby('movieId').agg({'rating': 'count'})
    if num_movies is not None:
        movies_to_keep = movies[(movies['rating'] >= num_ratings)].sample(
            n=num_movies, random_state=12).index
    else:
        movies_to_keep = movies[(movies['rating'] >= num_ratings)].index
    data = data[data['movieId'].isin(movies_to_keep)]
    # take a random sample of size `num_ratings` for each movie
    data = data.groupby('movieId').sample(n=num_ratings, random_state=42)
    # shuffle rows to randomize data stream
    data = data.sample(frac=1, random_state=42)
    # reset index to create pseudo-timestamp index
    data = data.reset_index(drop=True)
    return data


def get_data(num_ratings, num_movies=None):
    data = read_data()
    data = preprocess_data(data, num_ratings, num_movies)
    return data


# Plot ################################################################

def plot_rewards(*policies, title=None):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(11,5))
    fig.suptitle(title)
    for policy in policies:
        # get cumulative rewards
        cumsum_rewards = policy.history.reward.cumsum()
        # get average rewards
        timesteps = np.arange(len(cumsum_rewards)) + 1
        avg_rewards = cumsum_rewards / timesteps
        # plots
        ax1.plot(timesteps, avg_rewards, label=policy.name)
        ax2.plot(timesteps, cumsum_rewards, label=policy.name)
    #
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.set_xlabel('time step')
    ax1.set_ylabel('average reward')
    ax1.legend(loc='lower right')
    #
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax2.set_xlabel('time step')
    ax2.set_ylabel('cumulative reward')
    ax2.legend(loc='lower right')
    #
    plt.tight_layout()
    plt.show()


def plot_action_values(*policies):
    fig, axs = plt.subplots(nrows=1, ncols=len(policies), figsize=(15,5), squeeze=False)
    fig.suptitle("Action scores")
    axs = axs.ravel()
    for i, policy in enumerate(policies):
        cbar = True if i == len(axs)-1 else False
        sns.heatmap(policy.scores_log.T, ax=axs[i], vmin=2.5, vmax=5, cmap='hot',
                    cbar=cbar, xticklabels=1000, yticklabels=False)
        axs[i].set_xlabel('time step')
        axs[i].title.set_text(policy.name)
    axs[0].set_ylabel('movieId')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = get_data(10000)
    # rating stats vizualization
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
    # histogram of rating expectation per movie
    mean_ = data.groupby('movieId').agg({'rating': 'mean'})
    mean_.hist(grid=False, bins=24, ax=axs[0])
    axs[0].set_title("rating expectation")
    # histogram of rating variance per movie
    var_ = data.groupby('movieId').agg({'rating': 'var'})
    var_.hist(grid=False, bins=24, ax=axs[1])
    axs[1].set_title("rating variance")
    # scatter plot of var_ against mean_
    axs[2].scatter(mean_, var_, alpha=0.5)
    axs[2].set_title("rating variance vs. rating expectation")
    #
    plt.tight_layout()
    plt.show()
