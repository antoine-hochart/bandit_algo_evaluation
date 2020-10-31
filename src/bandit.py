import numpy as np
import pandas as pd
from scipy.stats import norm


class ReplayBandit():
    """ Implementation of a bandit problem with replay evaluation """
    def __init__(self, logged_events, batch_size=1):
        self.events = logged_events.rename(columns={'rating': 'reward'})
        self.actions = np.sort(logged_events['movieId'].unique())
        self.batch_size = batch_size
        self.stream_length = len(self.events) // batch_size
    
    def get_rewards(self, recommendations, n_event):
        # generate events
        idx = n_event * self.batch_size
        events =  self.events.iloc[idx:idx+self.batch_size]
        # keep only events that match with the recommendation slate
        rewards = events[events['movieId'].isin(recommendations)]
        return rewards


class SimulatedBandit():
    """ Implemenatation of a simulated bandit problem.
        Each action is chosen randomly in a slate of recommendations. """
    def __init__(self, logged_events, batch_size=1):
        self.actions = np.sort(logged_events['movieId'].unique())
        self.batch_size = batch_size
        # create series with multiindexing ('movieId', 'rating') and empirical freq for each rating
        self.proba = logged_events.value_counts(subset=['movieId', 'rating']) 
        self.proba = self.proba.div(logged_events['movieId'].value_counts(), level=0)
    
    def get_rewards(self, recommendations, n_event):
        # for each event in a batch, draw a random recommendation
        actions = np.random.choice(recommendations, size=self.batch_size, replace=True)
        # for each action, draw a rating/reward at random according to empirical frequencies
        ratings = [
            np.random.choice(self.proba.loc[movieId].index, size=1, p=self.proba.loc[movieId])[0]
                for movieId in actions ]
        rewards = pd.DataFrame({ 'movieId': actions, 'reward': ratings})
        return rewards


class SimulatedOptimisticBandit():
    """ Implementation of a simulated bandit problem.
        The best recommendation in the slate is selected. """
    def __init__(self, logged_events, batch_size=1):
        self.actions = np.sort(logged_events['movieId'].unique())
        self.batch_size = batch_size
        # create series with multiindexing ('movieId', 'rating') and probability for each rating
        self.proba = logged_events.value_counts(subset=['movieId', 'rating']) 
        self.proba = self.proba.div(logged_events['movieId'].value_counts(), level=0)
    
    def get_rewards(self, recommendations, n_event):
        # draw a random reward for each recommendation and each event in the batch
        events = [
            [ np.random.choice(self.proba.loc[rec].index, size=1, p=self.proba.loc[rec])[0]
                for rec in recommendations ]
            for _ in range(self.batch_size) ]
        # for each event in the batch, identify all recommendations with best rating
        best_actions = [
            np.argwhere(ratings == np.max(ratings)).ravel() for ratings in events]
        # choose a random best recommendation for each event in the batch
        actions = [np.random.choice(recs, 1)[0] for recs in best_actions]
        rewards = pd.DataFrame({
            'movieId': [recommendations[a] for a in actions],
            'reward': [events[i][a] for i, a in enumerate(actions)]})
        return rewards
        
