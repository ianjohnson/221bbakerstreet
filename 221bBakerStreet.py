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
import random


# The 221B Baker Street board
#  D - destination square,
#  N - normal square
#  S - start square,
#  \space - no square
baker_street_board = [[ "N", "N", "N", "N", "N", "N", "N", "N", "D", "N", "N", "N", "Ddocks", " ", " ", " ", " ", " ", " ", " " ],
                      [ "N", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "N", " ", " ", " ", " ", " ", " ", " ", " " ],
                      [ "D", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "N", " ", " ", " ", " ", " ", " ", " ", " " ],
                      [ "N", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "N", " ", " ", " ", " ", " ", " ", " ", " " ],
                      [ "N", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "N", " ", " ", " ", " ", " ", " ", " ", " " ],
                      [ "N", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "N", " ", " ", " ", " ", " ", " ", " ", " " ],
                      [ "N", "N", "N", "N", "D", "N", "N", "N", "N", "N", "N", "N", "N", "N", "N", "N", "N", "Ddocks", "N", "N" ],
                      [ " ", "N", " ", " ", " ", " ", "N", " ", " ", " ", " ", " ", " ", "N", " ", " ", " ", " ", " ", "N" ],
                      [ " ", "N", " ", " ", " ", " ", "N", " ", " ", " ", " ", " ", " ", "N", " ", " ", " ", " ", " ", "N" ],
                      [ " ", "D", " ", " ", " ", " ", "D", " ", " ", " ", " ", " ", " ", "N", " ", " ", " ", " ", " ", "N" ],
                      [ " ", "N", " ", " ", " ", " ", "N", " ", " ", " ", " ", " ", " ", "D", " ", " ", " ", " ", " ", "N" ],
                      [ " ", "N", " ", " ", " ", " ", "N", " ", " ", " ", " ", " ", " ", "N", " ", " ", " ", " ", " ", "N" ],
                      [ " ", "N", " ", " ", " ", " ", "N", " ", " ", " ", " ", " ", " ", "N", " ", " ", " ", " ", " ", "N" ],
                      [ " ", "D", "N", "N", "N", "N", "N", "N", "N", "D", "N", "N", "N", "N", "N", "N", "N", "N", "D", "N" ],
                      [ " ", "N", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "N", " ", " ", " ", " " ],
                      [ " ", "N", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "N", " ", " ", " ", " " ],
                      [ " ", "N", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "N", " ", " ", " ", " " ],
                      [ " ", "S", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "N", " ", " ", " ", " " ],
                      [ " ", "N", "N", "N", "N", "N", "D", "Dpark", "N", "N", "N", "D", "N", "N", "N", "D", " ", " ", " ", " " ],
                      [ " ", "N", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "N", " ", " ", " ", " " ],
                      [ " ", "N", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "N", " ", " ", " ", " " ],
                      [ " ", "Dpark", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "Dpark", "N", " ", " ", " ", " " ]]


class Action(object):
  def __init__(self, square):
    self.__square = square

  @property
  def square(self):
    return self.__square

class North(Action):
  def __init__(self, square):
    super(North, self).__init__(square)

  def __str__(self):
    return 'N'

class East(Action):
  def __init__(self, square):
    super(East, self).__init__(square)

  def __str__(self):
    return 'E'

class South(Action):
  def __init__(self, square):
    super(South, self).__init__(square)

  def __str__(self):
    return 'S'

class West(Action):
  def __init__(self, square):
    super(West, self).__init__(square)

  def __str__(self):
    return 'W'

  
class Square(object):
  def __init__(self, x, y):
    super(Square, self).__init__()
    self.__actions = list()
    self.__x = x
    self.__y = y

  @property
  def x(self):
    return self.__x

  @property
  def y(self):
    return self.__y

  @property
  def actions(self):
    return self.__actions

  @actions.setter
  def actions(self, acts):
    self.__actions = acts

  @property
  def is_destination(self):
    return False

  def next_square(self, action_class):
    matched_action = list(filter(lambda a: isinstance(a, action_class), self.actions))
    if len(matched_action) == 1:
      return matched_action[0].square
    return self

  def __str__(self):
    return "□ <" + ", ".join([str(a) for a in self.actions]) + " >[%d,%d]" % (self.x, self.y)

class DestinationSquare(Square):
  def __init__(self, x, y, group):
    super(DestinationSquare, self).__init__(x, y)
    self.__group = group

  @property
  def group(self):
    return self.__group

  @property
  def is_destination(self):
    return True

  def __str__(self):
    return "D□{" + self.group + "}<" + ", ".join([str(a) for a in self.actions]) + ">[%d,%d]" % (self.x, self.y)

class MultiDestinationSquare(DestinationSquare):
  def __init__(self, x, y, group):
    super(MultiDestinationSquare, self).__init__(x, y, group)

class StartSquare(Square):
  def __init__(self, x, y):
    super(StartSquare, self).__init__(x, y)

  def __str__(self):
    return "S□ <" + ", ".join([str(a) for a in self.actions]) + ">[%d,%d]" % (self.x, self.y)
        

class BakerStreetBoard(object):
  __GROUP_ID = 0

  def parse_board(board_array):
    start_square = None
    destinations = dict()
    action_board = list()

    def get_actions(board_array, x, y):
      actions = list()
      if y > 0 and board_array[y - 1][x] != ' ':
        actions.append(North(action_board[y - 1][x]))
      if y < len(board_array) - 1 and board_array[y + 1][x] != ' ':
        actions.append(South(action_board[y + 1][x]))
      if x > 0 and board_array[y][x - 1] != ' ':
        actions.append(West(action_board[y][x - 1]))
      if x < len(board_array[0]) - 1 and board_array[y][x + 1] != ' ':
        actions.append(East(action_board[y][x + 1]))
      return actions

    for y in range(0, len(board_array)):
      actions = list()
      for x in range(0, len(board_array[0])):
        square = board_array[y][x]
        if square == ' ':
          actions.append(None)
        elif square == 'N':
          actions.append(Square(x, y))
        elif square.startswith('D'):
          group_id = square[1:]
          if not group_id:
            group_id = str(BakerStreetBoard.__GROUP_ID)
            BakerStreetBoard.__GROUP_ID += 1
            destination = DestinationSquare(x, y, group_id)
          else:
            destination = MultiDestinationSquare(x, y, group_id)
          if destination.group in destinations:
            destinations[destination.group]['squares'].append(destination)
          else:
            destinations[destination.group] = {'visited' : False, 'squares' : [destination]}
          actions.append(destination)
        elif square == 'S':
          start_square = StartSquare(x, y)
          actions.append(start_square)
      action_board.append(actions)

    for y in range(0, len(board_array)):
      actions = list()
      for x in range(0, len(board_array[0])):
        square = board_array[y][x]
        if square != ' ':
          actions = get_actions(board_array, x, y)
          action_board[y][x].actions = actions

    return action_board, start_square, destinations

  def __init__(self, board_array):
    ab, ss, ds = BakerStreetBoard.parse_board(board_array)
    self.__board = ab
    self.__start_square = ss
    self.__destinations = ds
    self.__current_square = ss
    self.__leaving_destination = False

  def initialise(self):
    self.__current_square = self.__start_square
    for dest_group in self.__destinations:
      self.__destinations[dest_group]['visited'] = False
    return self.__current_square

  @property
  def board(self):
    return self.__board

  @property
  def dimensions(self):
    return (len(self.__board[0]), len(self.__board))

  @property
  def start_square(self):
    return self.__start_square

  @property
  def all_destinations_visited(self):
    not_visited = list(filter(lambda dest_group: self.__destinations[dest_group]['visited'] == False, self.__destinations))
    return False if len(not_visited) else True

  @property
  def current_square(self):
    return self.__current_square

  def next_square(self, action_class):
    # Park and Docks have multiple entrances/exits. Choose randomly how to leave.
    if isinstance(self.__current_square, MultiDestinationSquare):
      if self.__leaving_destination:
        next_square = self.__current_square.next_square(action_class)
        if next_square != self.__current_square:
          self.__current_square = next_square
          self.__leaving_destination = False
      else:
        idx = random.randint(0, len(self.__destinations[self.__current_square.group]['squares']) - 1)
        self.__current_square = self.__destinations[self.__current_square.group]['squares'][idx]
        self.__leaving_destination = True
    else:
      self.__current_square = self.__current_square.next_square(action_class)
      # Record a visit to a destination square
      if self.__current_square.is_destination:
        self.__destinations[self.__current_square.group]['visited'] = True

    return self.__current_square

  def __str__(self):
    row_strs = list()
    for row in self.__board:
      row_strs.append(', '.join([str(e) for e in row]))
    return '\n'.join(row_strs)


class BakerStreetEnvironment(object):
  def __init__(self, board):
    super().__init__()
    self.__board = board
    self.__actions = [North, East, South, West]
    self.__visited_squares = set()

  @property
  def states(self):
    return self.__board.board

  @property
  def actions(self):
    return self.__actions

  def random_action(self):
    return np.random.choice(self.__actions)

  def reset(self):
    self.__visited_squares = set()
    return self.__board.initialise()

  def execute(self, action):
    current_square = self.__board.current_square
    next_square = self.__board.next_square(action)
    terminal = self.__board.all_destinations_visited and next_square == self.__board.start_square
    reward = 100 if terminal else -1 if next_square not in self.__visited_squares else -2
    self.__visited_squares.add(next_square)
    if current_square == next_square:
      reward = -1000
    return next_square, terminal, reward

  def render(self):
    for row in self.__board.board:
      for col in row:
        if col:
          print("□", end = '')
          if isinstance(col, DestinationSquare):
            print(" {%2s}" % col.group[0:2], end = '')
          elif isinstance(col, StartSquare):
            print(" ⌂   ", end = '')
          else:
            print("     ", end = '')
        else:
          print("======", end = '')
      print()


def max_action(Q, state, actions):
  values = np.array([Q[state,a] for a in actions])
  action = np.argmax(values)
  return actions[action]

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

  board = BakerStreetBoard(baker_street_board)
  environment = BakerStreetEnvironment(board)
  environment.render()

  Q = {}
  for square in [item for sublist in environment.states for item in sublist if item]:
    for action in environment.actions:
      Q[square, action] = random.uniform(-1.0, 1.0)

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
      new_square, finished, reward = environment.execute(action)
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
