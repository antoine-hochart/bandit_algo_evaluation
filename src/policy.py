from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.stats import beta


# Abstract Base Policy ###############################################

class ABPolicy(ABC):
    def __init__(self, bandit, slate_size, scores_logging):
        self.name = None
        self.slate_size = slate_size
        self.history = pd.DataFrame(data=None, columns=['movieId', 'reward'])
        if scores_logging is True:
            self.scores_log = pd.DataFrame(data=None, columns=bandit.actions)
        else:
            self.scores_log = None

    @abstractmethod
    def get_recommendations(self):
        ...

    def update(self, rewards):
        # append new events to history
        self.history = self.history.append(rewards, ignore_index=True)
    
    def _sort_actions(self, scores):
        """ Sort actions by score and shuffle actions with same score
            Inputs:
                scores: pandas.Series with actions as index """
        sorted_actions = sorted(
            scores.sample(frac=1).index,
            key=lambda idx: scores.loc[idx],
            reverse=True)
        return sorted_actions

    def _update_scores_history(self, scores):
        if self.scores_log is not None:
            self.scores_log = self.scores_log.append(
                pd.DataFrame(
                    data=scores.to_numpy().reshape((1,-1)),
                    columns=self.scores_log.columns),
                ignore_index=True)
            self.scores_log = self.scores_log.astype('float')



# Epsilon Greedy Policy ##############################################

class EpsilonGreedy(ABPolicy):
    def __init__(self, bandit, epsilon, slate_size=1, scores_logging=False):
        super(EpsilonGreedy, self).__init__(bandit, slate_size, scores_logging)
        self.name = '{}-Greedy'.format(epsilon)
        self.epsilon = epsilon
        self.action_values = pd.DataFrame(data=0, columns=['value', 'count'],
                                          index=bandit.actions)

    def get_recommendations(self):
        # sort actions by value and shuffle actions with same value
        sorted_actions = self._sort_actions(self.action_values['value'])
        # choose recommendations
        if np.random.random() < self.epsilon:
            recs = np.random.choice(sorted_actions[self.slate_size:],
                                    size=self.slate_size, replace=False)
        else:
            recs = sorted_actions[:self.slate_size]
        # update history of action scores
        self._update_scores_history(self.action_values['value'])
        return recs

    def update(self, rewards):
        super(EpsilonGreedy, self).update(rewards)
        # update action values
        for _, (movieId, reward) in rewards.iterrows():
            value = self.action_values.loc[movieId, 'value']
            N = self.action_values.loc[movieId, 'count']
            self.action_values.loc[movieId, 'value'] = (value * N + reward) / (N + 1)
            self.action_values.loc[movieId, 'count'] += 1
    

# Upper Confidence Bound Policy ######################################

class UCB1(ABPolicy):
    def __init__(self, bandit, slate_size=1, scores_logging=False):
        super(UCB1, self).__init__(bandit, slate_size, scores_logging)
        self.name = 'UCB1'
        self.action_values = pd.DataFrame(data=0, columns=['value', 'count'],
                                          index=bandit.actions)

    def get_recommendations(self):
        # compute UCB for each action
        current_step = len(self.history)
        if current_step > 0:
            scores = self.action_values['count'].apply(
                lambda N: np.sqrt(2*np.log(current_step) / N) if N > 0 else np.Inf)
            scores = scores + self.action_values['value']
        else:
            scores = pd.Series(data=np.Inf, index=self.action_values.index)
        # sort actions by score and shuffle actions with same score
        sorted_actions = self._sort_actions(scores)
        # choose recommendations
        recs = sorted_actions[:self.slate_size]
        # update history of action scores
        self._update_scores_history(scores)
        return recs

    def update(self, rewards):
        super(UCB1, self).update(rewards)
        # update action values
        for _, (movieId, reward) in rewards.iterrows():
            value = self.action_values.loc[movieId, 'value']
            N = self.action_values.loc[movieId, 'count']
            self.action_values.loc[movieId, 'value'] = (value * N + reward) / (N + 1)
            self.action_values.loc[movieId, 'count'] += 1


# Thompson Sampling Policy ###########################################

class TS(ABPolicy):
    def __init__(self, bandit, slate_size=1, scores_logging=False):
        super(TS, self).__init__(bandit, slate_size, scores_logging)
        self.name = 'Thompson Sampling'
        self.beta_params = pd.DataFrame(data=1, columns=['alpha', 'beta'],
                                        index=bandit.actions)
    
    def get_recommendations(self):
        # sample expected value for each action
        expected_values = pd.Series(
            data=4.5 * beta.rvs(self.beta_params['alpha'], self.beta_params['beta']) + 0.5,
            index=self.beta_params.index)
        # sort actions by value and shuffle actions with same value
        sorted_actions = self._sort_actions(expected_values)
        # choose recommendations
        recs = sorted_actions[:self.slate_size]
        # update history of action scores
        self._update_scores_history(expected_values)
        return recs

    def update(self, rewards):
        super(TS, self).update(rewards)
        # update action value distribution prior
        for _, (movieId, reward) in rewards.iterrows():
            self.beta_params.loc[movieId, 'alpha'] += (reward - 0.5) / 4.5
            self.beta_params.loc[movieId, 'beta'] += (5.0 - reward) / 4.5
        
