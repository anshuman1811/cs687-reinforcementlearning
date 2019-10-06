import numpy as np
from typing import Tuple
from .skeleton import Environment


class Cartpole(Environment):
    """
    The cart-pole environment as described in the 687 course material. This
    domain is modeled as a pole balancing on a cart. The agent must learn to
    move the cart forwards and backwards to keep the pole from falling.

    Actions: left (0) and right (1)
    Reward: 1 always

    Environment Dynamics: See the work of Florian 2007
    (Correct equations for the dynamics of the cart-pole system) for the
    observation of the correct dynamics.
    """

    def __init__(self):
        # TODO: properly define the variables below
        self._name = "Cartpole"
        self._action = None
        self._reward = 0.0
        self._isEnd = False
        self._gamma = 1.0

        # define the state # NOTE: you must use these variable names
        self._x = 0.  # horizontal position of cart
        self._v = 0.  # horizontal velocity of the cart
        self._theta = 0.  # angle of the pole
        self._dtheta = 0.  # angular velocity of the pole
        self._state = np.array([self._x, self._v, self._theta, self._dtheta])

        # dynamics
        self._g = 9.8  # gravitational acceleration (m/s^2)
        self._mp = 0.1  # pole mass
        self._mc = 1.0  # cart mass
        self._l = 0.5  # (1/2) * pole length
        self._dt = 0.02  # timestep
        self._t = 0  # total time elapsed  NOTE: USE must use this variable
        self.Fmag = 10.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def action(self) -> int:
        return self._action

    @property
    def isEnd(self) -> bool:
        return self._isEnd

    @property
    def state(self) -> np.ndarray:
        return np.array([self._x, self._v, self._theta, self._dtheta])

    def nextState(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute the next state of the pendulum using the euler approximation to the dynamics
        """
        # print (self._t, action)
        F = self.Fmag
        if action == 0:
            F = -self.Fmag
        dx = self._v
        dtheta = self._dtheta
        domega = (self._g * np.sin(self._theta) + np.cos(self._theta) * (-F - self._mp * self._l * np.square(self._dtheta) * np.sin(self._theta)) / (self._mc + self._mp) ) / (self._l * (4.0/3.0 - (self._mp * np.square(np.cos(self._theta)) / (self._mc + self._mp)) ) )
        dv = (F + (self._mp * self._l)*(np.square(dtheta)*np.sin(self._theta) - domega * np.cos(self._theta))) / (self._mc + self._mp)

        self._x = self._x + self._dt * dx
        self._v = self._v + self._dt * dv
        self._theta = self._theta + self._dt * dtheta
        self._dtheta = self._dtheta + self._dt * domega
        self._state = np.array([self._x, self._v, self._theta, self._dtheta])
        # print (self._state)

        return self._state

    def R(self, state: np.ndarray, action: int, nextState: np.ndarray) -> float:
        return 1.0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        takes one step in the environment and returns the next state, reward, and if it is in the terminal state
        """
        # print("Stepping")
        oldState = self._state
        nextState = self.nextState(self._state, action)

        # print (oldState, nextState)    
        self._t += self._dt
        self._isEnd = self.terminal()
        self._action = action

        reward = self.R (oldState, action, nextState)
        self._reward += reward
        
        # print("Current Accumulated reward = ", self._reward)
        return (nextState, reward, self._isEnd)

    def reset(self) -> None:
        """
        resets the state of the environment to the initial configuration
        """
        self.__init__()

    def terminal(self) -> bool:
        """
        The episode is at an end if:
            time is greater that 20 seconds
            pole falls |theta| > (pi/12.0)
            cart hits the sides |x| >= 3
        """
        # print (self._t, self._x, self._theta, (self._t > 20 or abs(self._x)>=3 or abs(self._theta) > np.pi/12.0))
        return (self._t >= 20 or abs(self._x) >= 3 or abs(self._theta) > np.pi/12.0)
