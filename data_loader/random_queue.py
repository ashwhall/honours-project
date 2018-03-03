import queue
import random

class RandomQueue(queue.Queue):
  def _put(self, item):
    n = len(self.queue)
    i = random.randint(0, n)
    self.queue.insert(i, item)
