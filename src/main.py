from tqdm import tqdm

from bandit import ReplayBandit, SimulatedBandit
from policy import EpsilonGreedy, UCB1, TS
from utils import get_data, plot_rewards, plot_action_values


######################################################################

NUM_RATINGS = 10000     # with full dataset  -> 10000
                        # with small dataset -> 30
NUM_MOVIES = None
SLATE_SIZE = 5
BATCH_SIZE = 100        # with replay eval   -> 100
                        # with simulated env -> 1
STREAM_LENGTH = 50000   # with full dataset  -> 50000
                        # with small dataset -> 150
MODE = 'replay'         # 'replay' or 'sim'
SCORES_LOG = False      # logging movie scores or not

######################################################################

# get data
logged_events = get_data(NUM_RATINGS, NUM_MOVIES)

# instantiate bandit problem
if MODE == 'replay':
    bandit = ReplayBandit(logged_events, BATCH_SIZE)
    STREAM_LENGTH = bandit.stream_length
    title="rewards for bandit problem with replay evaluation"
elif MODE == 'sim':
    bandit = SimulatedBandit(logged_events, BATCH_SIZE)
    title="rewards for bandit problem with simulated environment"

print("NUMBER OF MOVIES/ACTIONS: {}".format(len(bandit.actions)))
print()

# instantiate policies
policies = [
    EpsilonGreedy(bandit, epsilon=0.1, slate_size=SLATE_SIZE, scores_logging=SCORES_LOG),
    UCB1(bandit, slate_size=SLATE_SIZE, scores_logging=SCORES_LOG),
    TS(bandit, slate_size=SLATE_SIZE, scores_logging=SCORES_LOG),
    ]

# evaluate policies
for policy in policies:
    print("POLICY: {}".format(policy.name))
    for i in tqdm(range(STREAM_LENGTH), ascii=True):
        recs = policy.get_recommendations()
        rewards = bandit.get_rewards(recs, i)
        policy.update(rewards)
    print("HISTORY LENGTH: {}".format(len(policy.history)))
    print()

# plot results
plot_rewards(*policies, title=title)
if SCORES_LOG is True:
    plot_action_values(*policies)
