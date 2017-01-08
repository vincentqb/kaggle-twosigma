import kagglegym
# import numpy as np
# import pandas as pd


def make_prediction(observation):
    """Make prediction."""
    target = observation.target
    return target


# Interface for code competition
env = kagglegym.make()

# Get initial observation
observation = env.reset()

# Load train dataset
train = observation.train
print("Train has {} rows".format(len(train)))

# Template target to predict
target = observation.target
print("Target columns: {}".format(target.columns))

while True:
    """Iterate over data."""

    # Check timestamp
    timestamp = observation.features["timestamp"][0]
    if not timestamp % 100:
        print("Timestamp: {}".format(timestamp))

    # Make prediction
    target = make_prediction(observation)

    # Submit predicted target, and get back updated observation
    observation, reward, done, info = env.step(target)

    # Done?
    if done:
        print("Public score: {}".format(info["public_score"]))
        break
