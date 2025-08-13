from typing import List
from collections import deque, namedtuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'history'])

class ReplayBuffer:
  def __init__(self,
               capacity:int):
    # FIFO buffer
    self.buffer = deque(maxlen=capacity)

  def push(self,
           state:List[int],
           action:int,
           reward:int,
           next_state:List[int],
           done:int,
           history
           ):
    self.buffer.append(Experience(state, action, reward, next_state, done, history))

  def sample(self, batch_size:int):
    return random.sample(self.buffer, batch_size)

  def __len__(self):
    return len(self.buffer)
