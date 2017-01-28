import kagglegym
import numpy as np
import pandas as pd


def remove_outliers(col):
    """Remove outliers from column."""

    # Ignore missing values
    col = col.dropna()

    # First quantile
    q_low = col.quantile(.25)
    q_high = col.quantile(.75)
    q_diff = q_high - q_low

    # Add buffer to quantile
    low = q_low - 1.5 * q_diff
    high = q_high + 1.5 * q_diff

    # Drop values outside range
    col[(col > high) | (col < low)] = np.nan

    return col


class Model():

    def __init__(self):
        """Select model to run"""

        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()

        excl = ['id', 'sample', 'y', 'timestamp']
        self.cols = [c for c in obs.features.columns if c not in excl]

        # self.cols = ['fundamental_23', 'fundamental_37', 'technical_19', 'technical_27']
        # self.cols = ['fundamental_23']  # -0.0029627241405782616
        # self.cols = ['fundamental_37']  # -0.004656196123714797
        # self.cols = ['technical_19']  # 0.006625206966729358
        self.cols = ['technical_27']  # 0.00775421908968341
        # self.cols = ['technical_19', 'technical_27'] # -0.0038563192572963526
        self.y_min = -0.0380067
        self.y_max = 0.0380636

    def clip(self, col):
        """Clip values outside of [y_min, y_max]."""

        too_high = col > self.y_max
        col[too_high] = self.y_max
        too_low = col < self.y_min
        col[too_low] = self.y_min

        return col

    def initial_fit(self, obs):
        """Initial training."""

        train = obs.train
        target = train['y']

        # Keep only certain columns
        features = train[self.cols]

        # Find and remove outliers in target
        # outliers = obs[remove_outliers(obs['y']).isnull()].index
        # obs.drop(outliers, inplace=True)
        # target.drop(outliers, inplace=True)

        # Save min/max for clipping
        self.y_min = min(target)
        self.y_max = max(target)

        # Fill missing values
        # https://www.kaggle.com/c/two-sigma-financial-modeling/discussion/26205
        features.dropna(how='all', axis=1, inplace=True)
        self.mean_values = features.mean(axis=0)

        # Fit model
        self.fit(features, target)

    def fit(self, features, target):
        """Further training."""
        features = features[self.cols]
        # target = self.clip(target)
        features.fillna(self.mean_values, inplace=True)
        self.model.fit(features, target)

    def predict(self, features):
        """Make prediction."""
        features = features[self.cols]
        features.fillna(self.mean_values, inplace=True)
        target = self.model.predict(features)
        # target = self.clip(target)
        target = pd.DataFrame({'y': target, 'id': obs.features['id']})
        return target


# Interface for code competition
env = kagglegym.make()
# Get column names
# excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME,
#         env.TARGET_COL_NAME, env.TIME_COL_NAME]
# Get initial observations
obs = env.reset()

# Create model and run initial fit
model = Model()
model.initial_fit(obs)

done = False
# rewards = []
while not done:
    """Iterate over data."""

    # Check timestamp
    timestamp = obs.features["timestamp"][0]
    if not timestamp % 100:
        print("Timestamp: {}".format(timestamp))

    # Make prediction
    target = model.predict(obs.features)

    # Submit predicted target, and get back updated obs
    obs, reward, done, info = env.step(target)

    # Reinforcement
    # Fit points that were predicted better?

    # rewards.append(reward)

print("Public score: {}".format(info["public_score"]))
