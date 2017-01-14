import kagglegym
# import numpy as np
# import pandas as pd


def make_prediction(obs):
    """Make prediction."""
    target = obs.target
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
# excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
# excl = ['id', 'sample', 'y', 'timestamp']
# col = [c for c in obs.train.columns if c not in excl]

while True:
    """Iterate over data."""

    # Check timestamp
    timestamp = obs.features["timestamp"][0]
    if not timestamp % 100:
        print("Timestamp: {}".format(timestamp))

    # Make prediction
    target = make_prediction(obs)

    # Submit predicted target, and get back updated obs
    obs, reward, done, info = env.step(target)

    # Done?
    if done:
        print("Public score: {}".format(info["public_score"]))
        break
