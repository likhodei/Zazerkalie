import numpy as np
from absl import app

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D

import quad
from path import RRTStar, Map
import trajectory

import enum
from collections import namedtuple
 
class Board:
    def __init__(self, range: int = 100):
        # 2D boxes   lx, ly, hx, hy
        obstacles = [[-5, 25, 20, 35],
                    [30, 25, 55, 35],
                    [45, 35, 55, 60],
                    [45, 75, 55, 85],
                    [-5, 65, 30, 70],
                    [70, 50, 80, 80]]

        # limits on map dimensions
        self.bounds = np.array([0, 100])

        # create map with obstacles
        self.mapobs = Map(obstacles, self.bounds, dim = 2)
        self.position = dict()

    def plan(self, start, goal, max_iter = 100): # plan a path from start to goal
        rrt = RRTStar(start, goal, self.mapobs, max_iter, goal_sample_rate = 0.02)
        return rrt.plan()

    def place(self, player, name, point):
        self.position[name] = point

class Point(namedtuple('Point', 'row col')):
    def foo(self):
        pass

    def __deepcopy__(self, memodict={}):
        # These are very immutable.
        return self

class Move:
    def __init__(self, pos, attitude = [0, 0, 0]):
        self.qrotor = quad.Rotor(pos, attitude)

    def setup(self, name, dt, U, M): # todo: refactory
        self.tag = name,
        self.U = U
        self.M = M
        self.dt = dt

    def get_state(self):
        return self.qrotor.get_state()

    def update(self, dt, U, M):
        self.qrotor.update(dt, U, M)

class Player(enum.Enum):
    black = 1
    white = 2

    @property
    def other(self):
        return Player.black if self == Player.white else Player.white

class Scene:
    def __init__(self, board: Board, next_player: Player, previous, move: Move):
        self.next_player = next_player
        self.done = False
        self.board = board
        self.t = 0

    @classmethod
    def new_simulation(cls, place_size):
        if isinstance(place_size, int):
            bound = (place_size, place_size)

        board = Board(bound)
        return Scene(board, Player.white, None, None)

    def is_over(self):
        return self.done

    def apply(self, mv: Move):
        snapshot = self.board # deep copy

        # todo: change
        U = mv.U
        M = mv.M
        dt = mv.dt

        mv.update(dt, U, M)

        point = mv.get_state()
        snapshot.place(self.next_player, mv.tag, point) # todo: test

        return Scene(snapshot, self.next_player.other, self, mv)

class Agent:
    def __init__(self, name: str, waypoints, cost, low=10, high=50):
        self.name = name 
        self.min_cost = cost
        self.max_velocity = 50
        self.control_frequency = 200

        p = np.array(waypoints, dtype=float)
        h = np.random.randint(low, high, p.shape[0])
        h = np.expand_dims(h, axis=1)
        self.waypoints = np.concatenate((p, h), axis=1)

        #Generate trajectory through waypoints
        self.traj = trajectory.Generator(self.waypoints, self.max_velocity, gamma = 1e6)
        self.Tmax = self.traj.TS[-1]

        self.dt = 1 / self.control_frequency
        self.mv = None

    def select(self, scene: Scene) -> Move:
        pos = scene.board.position.get(self.name, None)
        if pos == None:
            self.mv = Move(self.traj.get_des_state(0).pos)

        s = self.mv.get_state()
        d = self.traj.get_des_state(scene.t)

        U, M = quad.Run(s, d)

        self.mv.setup(self.name, self.dt, U, M)
        return self.mv

    def diagnostics(self):
        return {}

def mvp(_):
    game = Scene.new_simulation(100)
    bots = {
        Player.black : Agent('A',*game.board.plan(np.array([80, 20]), np.array([40, 37]))),
        Player.white : Agent('B',*game.board.plan(np.array([80, 20]), np.array([10, 80])))
    }

    while not game.is_over():
        move = bots[game.next_player].select(game)
        game = game.apply(move)
        break

    pass

if __name__ == "__main__":
    app.run(mvp)
