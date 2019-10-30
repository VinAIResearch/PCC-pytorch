"""Get Cartpole Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import numpy as np
from pathlib import Path

root_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
os.sys.path.append(root_path)

from mdp import cartpole_mdp


def get_cartpole_dataset(rootpath,
                         num_samples=15000,
                         width=80,
                         height=80,
                         frequency=50,
                         noise=0.0,
                         overwrite_datasets=True,
                         random_sample_states=True,
                         **kwargs):
  """Create a dataset for the visual cart-pole balance MDP.

  For the problem to at least approximately have the Markov property, we need at
  least two images to be able to estimate the velocity.

  Args:
    rootpath: root path of the dataset.
    num_samples: number of samples to generate for the dataset.
    width: integer indicating the width of the rendered image.
    height: integer indicating the height of the rendered image.
    frequency: float indicating the simulator frequency (discrete steps).
    noise: float indicating the magnitude of additive noise to the transitions.
    overwrite_datasets: boolean, whether to overwrite existing dataset.
    random_sample_states: whether data collection occurs randomly or sequently.
    **kwargs: additional keyed arguments.

  Returns:
    A dateset of tuples of interactions for the MDP
    {(x_t, u_t, x_t+1, s_t, s_t+1)}.
  """
  # Unused.
  del kwargs

  # Currently only supporting random sampling.
  assert random_sample_states

  # Path to dump date or read data from.
  filename_suffix = 'cartpole_{num_samples}_{frequency}_{noise}'.format(
      num_samples=num_samples, frequency=frequency,
      noise=str(noise).replace('.', '-'))
  filename = os.path.join(rootpath, filename_suffix)

  if overwrite_datasets:
    print('Creating cartpole dataset', filename)
    mdp = cartpole_mdp.VisualCartPoleBalance(height=height, width=width,
                                             frequency=frequency, noise=noise)

    # Data buffers to fill.
    x_data = np.zeros((num_samples, width, height, 2), dtype='float32')
    u_data = np.zeros((num_samples, mdp.action_dim), dtype='float32')
    x_next_data = np.zeros((num_samples, width, height, 2), dtype='float32')
    state_data = np.zeros((num_samples, 4, 2), dtype='float32')
    state_next_data = np.zeros((num_samples, 4, 2), dtype='float32')

    # Generate interaction tuples (random states and actions).
    for sample in range(num_samples):
      s0 = mdp.sample_random_state()
      a0 = np.atleast_1d(
          np.random.uniform(mdp.avail_force[0], mdp.avail_force[1]))
      s1 = mdp.transition_function(s0, a0)
      a1 = np.atleast_1d(
          np.random.uniform(mdp.avail_force[0], mdp.avail_force[1]))
      s2 = mdp.transition_function(s1, a1)

      ## Store interaction tuple.
      # Current state (w/ history).
      x_data[sample, :, :, 0] = s0[1][:, :, 0]
      x_data[sample, :, :, 1] = s1[1][:, :, 0]
      state_data[sample, :, 0] = s0[0][0:4]
      state_data[sample, :, 1] = s1[0][0:4]
      # Action.
      u_data[sample] = a1
      # Next state (w/ history).
      x_next_data[sample, :, :, 0] = s1[1][:, :, 0]
      x_next_data[sample, :, :, 1] = s2[1][:, :, 0]
      state_next_data[sample, :, 0] = s1[0][0:4]
      state_next_data[sample, :, 1] = s2[0][0:4]

    with gfile.Open(filename, 'wb') as f:
      pickle.dump((x_data, u_data, x_next_data, state_data, state_next_data), f)

    return x_data, u_data, x_next_data, state_data, state_next_data

  print('Loading cartpole dataset', filename)
  return pickle.load(gfile.Open(filename, 'rb'))
