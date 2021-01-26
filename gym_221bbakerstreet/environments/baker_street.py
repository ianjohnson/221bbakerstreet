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
import gym
import numpy as np


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
        idx = np.random.randint(0, len(self.__destinations[self.__current_square.group]['squares']) - 1)
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


class BakerStreetEnvironment(gym.Env):
  metadata = {'render.modes' : ['human']}

  def __init__(self):
    super().__init__()
    self.__board = BakerStreetBoard(baker_street_board)
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
    return next_square, terminal, reward, {}

  def render(self, mode = 'human'):
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

    
