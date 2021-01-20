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

  def next_square(self, action):
    matched_action = list(filter(lambda a: a == action, self.actions))
    if len(matched_action) == 1:
      return matched_action[0].square
    return self

  def __str__(self):
    return "□ <" + ", ".join([str(a) for a in self.actions]) + ">"

class DestinationSquare(Square):
  def __init__(self, x, y, group):
    super(DestinationSquare, self).__init__(x, y)
    self.__group = group

  @property
  def group(self):
    return self.__group

  def __str__(self):
    return "D□{" + self.group + "}<" + ", ".join([str(a) for a in self.actions]) + ">"

class StartSquare(Square):
  def __init__(self, x, y):
    super(StartSquare, self).__init__(x, y)

  def __str__(self):
    return "S□ <" + ", ".join([str(a) for a in self.actions]) + ">"
        

class Board(object):
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
          if not square[1:]:
            group_id = str(Board.__GROUP_ID)
            Board.__GROUP_ID += 1
          destination = DestinationSquare(x, y, group_id)
          if destination.group in destinations:
            destinations[destination.group].append(destination)
          else:
            destinations[destination.group] = [destination]
          actions.append(destination)
        elif square == 'S':
          actions.append(StartSquare(x, y))
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
    ab, ss, ds = Board.parse_board(board_array)
    self.__board = ab
    self.__start_square = ss
    self.__destinations = ds
    self.__current_square = ss

  def initialise(self):
    self.__current_square = self.__start_square

  @property
  def start_square(self):
    return self.__start_square

  @property
  def destinations(self):
    return self.__destinations

  def next_square(self, action):
    self.__current_square = self.__current_square.next_square(action)
    return self.__current_square

  def __str__(self):
    row_strs = list()
    for row in self.__board:
      row_strs.append(', '.join([str(e) for e in row]))
    return '\n'.join(row_strs)
        

if __name__ == '__main__':
  board = Board(baker_street_board)
  print(board)
