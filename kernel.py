import kagglegym
import numpy as np
# import pandas as pd


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

    def __init__(self, cols):
        """Select model to run"""

        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()

        self.cols = cols

    def clip(self, col):
        """Clip values outside of [y_min, y_max]."""

        too_high = col > self.y_max
        col[too_high] = self.y_max
        too_low = col < self.y_min
        col[too_low] = self.y_min

        return col

    def initial_fit(self, obs, target):
        """Initial training."""

        # Keep only certain columns
        obs = obs[self.cols]

        # Find and remove outliers in target
        # outliers = obs[remove_outliers(obs['y']).isnull()].index
        # obs.drop(outliers, inplace=True)
        # target.drop(outliers, inplace=True)

        # Save min/max for clipping
        self.y_min = min(target)
        self.y_max = max(target)

        # Fill missing values
        self.mean_values = obs.mean(axis=0)
        obs.fillna(self.mean_values, inplace=True)

        # Fit model
        self.model.fit(obs, target)

    def fit(self, obs, target):
        """Further training."""
        obs = obs[self.cols]
        # target = self.clip(target)
        obs.fillna(self.mean_values, inplace=True)
        self.model.fit(obs, target)

    def predict(self, obs):
        """Make prediction."""
        obs = obs[self.cols]
        obs.fillna(self.mean_values, inplace=True)
        target = self.model.predict(obs)
        # target = self.clip(target)
        return target


# Interface for code competition
env = kagglegym.make()

# Get initial obs
obs = env.reset()

# Load train dataset
train = obs.train
print("Train has {} rows".format(len(train)))

# Template target to predict
target = obs.target
print("Target columns: {}".format(target.columns))

# Excluse some columns
# print(dir(env))
excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME,
        env.TARGET_COL_NAME, env.TIME_COL_NAME]
# excl = ['id', 'sample', 'y', 'timestamp']
# col = [c for c in obs.train.columns if c not in excl]

# Create model and run initial fit
model = Model()
model.initial_fit(train, target)

done = False
# rewards = []
while not done:
    """Iterate over data."""

    # Check timestamp
    timestamp = obs.features["timestamp"][0]
    if not timestamp % 100:
        print("Timestamp: {}".format(timestamp))

    # Make prediction
    target = model.predict(obs)

    # Submit predicted target, and get back updated obs
    obs, reward, done, info = env.step(target)

    # Reinforcement
    # Fit points that were predicted better?

    # rewards.append(reward)

print("Public score: {}".format(info["public_score"]))
