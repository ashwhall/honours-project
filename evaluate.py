import contextlib
import os
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt

from constants import Constants
from cached_outputs import CachedOutputs
import data_loader.data_partitioner as DataPartitioner

class Evaluate:
  def __init__(self, config_file):
    Constants.load_config(config_file)

  def _plot_results(self, num_ways, top_ones):
    plt.plot(num_ways, top_ones, 'b-+')
    plt.axis([0, np.max(num_ways), 0, 1])
    plt.ylabel("Top-1 (%)")
    plt.xlabel("N-Way")
    plt.title("Generalisation")
    plt.show()

  def evaluate_all(self):
    results = CachedOutputs.load('model_outputs')
    num_ways = []
    top_ones = []

    print("|_N-Shot_|_N-Way_|__Top-1_|")
    for num_way, result in sorted(results.items(), key=lambda x: x[0]):
      # print("Num classes: {}".format(num_way))
      predictions = np.asarray(result['predictions'])
      targets = np.asarray(result['targets'])
      losses = np.asarray(result['losses'])

      top_one = np.mean(np.argmax(predictions, -1) == targets)
      print("|  {:>4d}  |  {:>3d}  | {:>6.2f} |".format(
          1, num_way, 100 * top_one
      ))
      num_ways.append(num_way)
      top_ones.append(top_one)
    self._plot_results(num_ways, top_ones)


def main(argv):
  np.set_printoptions(precision=4, linewidth=150, suppress=True)

  if '--config_file' in argv:
    config_file = argv[argv.index('--config_file') + 1]
  else:
    config_file = 'basic_config.yml'

  evaluator = Evaluate(config_file)
  evaluator.evaluate_all()


if __name__ == "__main__":
  main(sys.argv)
