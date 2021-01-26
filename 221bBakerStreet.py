#!/usr/bin/python3
#
# The MIT License
#
# Copyright (c) 2010-2020 Google LLC. http://angularjs.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
import matplotlib.pyplot as plt
import numpy as np
import gym
import gym_221bbakerstreet

from gym_221bbakerstreet.environments.baker_street import North, East, South, West


def show_actions(Q, environment):
  for row in environment.states:
    for col in row:
      if col:
        values = np.array([Q[col, a] for a in environment.actions])
        action = np.argmax(values)
        klass = environment.actions[action]
        if klass == North:
          print("↑ ", end = '')
        elif klass == East:
          print("→ ", end = '')
        elif klass == South:
          print("↓ ", end = '')
        else:
          print("← ", end = '')
      else:
        print("  ", end = '')
    print()


def max_action(Q, state, actions):
  values = np.array([Q[state,a] for a in actions])
  action = np.argmax(values)
  return actions[action]


if __name__ == '__main__':
  import argparse
  import os

  default_no_trials = 50000
  default_no_exploration_trials = 45000
  default_max_no_steps = 1e4
  default_every = 1000
  default_learning_rate = 1e-3
  default_discount = 1.0 - (1.0 / default_max_no_steps)
  default_epsilon = 1.0

  parser = argparse.ArgumentParser(prog = "221bBakerStreet",
                                   description = "221b Baker Street board route finder")
  parser.add_argument('--trials', '-t',
                      dest = 'no_trials',
                      type = int,
                      default = default_no_trials,
                      help = "Number of trials (default: %s)" % default_no_trials)
  parser.add_argument('--etrials', '-e',
                      dest = 'no_exploration_trials',
                      type = int,
                      default = default_no_exploration_trials,
                      help = "Number of exploration trials (default: %s)" % default_no_exploration_trials)
  parser.add_argument('--steps', '-s',
                      dest = 'max_no_steps',
                      type = int,
                      default = default_max_no_steps,
                      help = "Maximum number of steps in a trial (default: %s)" % default_max_no_steps)
  parser.add_argument('--every', '-v',
                      dest = 'every_trials',
                      type = int,
                      default = default_every,
                      help = "Display results every N trials (default: %s)" % default_every)
  parser.add_argument('--learningrate', '-r',
                      dest = 'learning_rate',
                      type = float,
                      default = default_learning_rate,
                      help = "Learning rate (default: %.3f)" % default_learning_rate)
  parser.add_argument('--discount', '-d',
                      dest = 'discount',
                      type = float,
                      default = default_discount,
                      help = "Discount (default: %.6f)" % default_discount)
  parser.add_argument('--epsilon', '-p',
                      dest = 'epsilon',
                      type = float,
                      default = default_epsilon,
                      help = "Initial exploration (default: %.3f)" % default_epsilon)
  args = parser.parse_args()

  environment = gym.make('BakerStreet-v1')
  environment.render()

  Q = {}
  for square in [item for sublist in environment.states for item in sublist if item]:
    for action in environment.actions:
      Q[square, action] = np.random.uniform(-1.0, 1.0)

  # Unpack parameters
  alpha = args.learning_rate
  discount = args.discount
  epsilon = args.epsilon
  no_trials = args.no_trials
  no_exploration_epochs = args.no_exploration_trials
  every = args.every_trials
  max_no_steps = args.max_no_steps

  total_rewards = []
  plt.gcf().canvas.set_window_title("221b Baker Street Board Route Finder (pid = %d)" % os.getpid())
  plt.xlabel("Trial Number")
  plt.ylabel("Trial Reward")
  plt.title("Trial Rewards")
  plt.plot(total_rewards, linewidth = 1, color = 'blue')
  plt.pause(0.05)
  for i in range(no_trials):
    if i % every == 0:
      show_actions(Q, environment)
      print('Epoch: ', i, end = '')
    finished = False
    epoch_rewards = 0
    current_square = environment.reset()
    no_steps = 0
    while not finished:
      rand = np.random.random()
      action = max_action(Q, current_square, environment.actions) if rand < (1-epsilon) else environment.random_action()
      new_square, finished, reward, info = environment.execute(action)
      epoch_rewards += reward
      new_action = max_action(Q, new_square, environment.actions)
      Q[current_square, action] += alpha*(reward + discount*Q[new_square, new_action] - Q[current_square, action])
      current_square = new_square
      no_steps += 1
      if no_steps > max_no_steps:
        finished = True

    if epsilon - (1.0 / no_exploration_epochs) > 0:
      epsilon -= 1.0 / no_exploration_epochs
    else:
      epsilon = 0

    if i % every == 0:
      print(" (%.2f)" % epoch_rewards)
      plt.plot(total_rewards, linewidth = 1, color = 'blue')
      plt.gcf().canvas.draw_idle()
      plt.gcf().canvas.start_event_loop(0.05)
    total_rewards.append(epoch_rewards)

  print("\nEnd")
  print("%.2f" % epoch_rewards)
  show_actions(Q, environment)

  plt.plot(total_rewards, linewidth = 1)
  plt.show()
