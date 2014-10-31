"""
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import lmfit
import numpy

class TooManyIterationsException(Exception):
  pass

class Explorer(object):
  """This class tries to "explore" the local minima of the error function.

  The error function is the sum of square of distance errors for known distances.

  Given an initial location near a local minima, it will attempt to descend the minima, and then evaluate all points on
  a grid of a fixed size (currently hard-coded to 1/32) in and around that local minima.
  """

  def __init__(self, connections, limit):
    self.values = {}
    self.connections = connections
    self.correct_locations = []
    self.limit = limit

  def explore(self, location):
    """Explore the local minima near location.

    This calculates the error function for all grid-aligned locations in and around the volume where error=0.
    Afterwards, the correct_locations field will be populated with all the grid-aligned locations where error=0.

    Raises:
      TooManyIterationsException: if more than [limit] (from the constructor) locations are calculated
    """
    self.generic_explore(location,
                         lambda location: self.explore_plane(location),
                         lambda params: self.objective(params),
                         3,
                         False)

  def minimize(self, initial_guess, objective, dimensions):
    params = lmfit.Parameters()
    params.add('x', value=initial_guess[0], vary=True)
    params.add('y', value=initial_guess[1], vary=False)
    params.add('z', value=initial_guess[2], vary=False)

    if dimensions > 1:
      params['y'].vary = True
    if dimensions > 2:
      params['z'].vary = True

    estimation = lmfit.minimize(objective, params)
    if estimation.success:
      return numpy.array([estimation.params['x'].value,
                          estimation.params['y'].value,
                          estimation.params['z'].value])
    return None

  def objective(self, params, x=None, y=None, z=None):
    """An objective function for use with lmfit's minimize function."""

    if x is None:
      x = params['x'].value
    if y is None:
      y = params['y'].value
    if z is None:
      z = params['z'].value
    guess = numpy.array([x, y, z])
    error = []

    for name, other_location, expected_distance in self.connections:
      error.append(self.calculate_single_error(guess, other_location, expected_distance) ** 2)
    return error

  def generic_explore(self, location, explore_func, objective_func, dimensions, exit_early=True):
    if self.get_error(location) != 0:
      minimum_location = self.minimize(location, objective_func, dimensions)

      minimum_value = self.get_error(minimum_location)
      if minimum_value > .0001 and exit_early:
        return False
    else:
      minimum_location = location

    initial_location = numpy.rint(minimum_location * 32) / 32
    next_location = initial_location
    explore_func(next_location)

    vector = numpy.array([0, 0, 0], float)
    vector[dimensions-1] = 1/32.0

    next_location = next_location + vector
    while explore_func(next_location):
      next_location = next_location + vector

    next_location = initial_location - vector
    while explore_func(next_location):
      next_location = next_location - vector

    return True

  def explore_line(self, location):
    return self.generic_explore(location,
                               lambda location: self.get_error(location) < .0001,
                               lambda params: self.objective(params, y=location[1], z=location[2]),
                               1)

  def explore_plane(self, location):
    return self.generic_explore(location,
                               lambda location: self.explore_line(location),
                               lambda params: self.objective(params, z=location[2]),
                               2)

  def get_error(self, location):
    """Gets the error value at the given location."""
    location_tuple = tuple(location)
    value = self.values.get(location_tuple)
    if value is not None:
      return value
    else:
      value = self.calculate_error(location)
      self.values[location_tuple] = value
      if len(self.values) > self.limit:
        raise TooManyIterationsException()
      return value

  def calculate_single_error(self, location, other_location, expected_distance):
    """Calculates the raw error for a single known distance."""

    # First, calculate the distance using 32-bit floats, to match the calculation used by the game
    float32_distance = numpy.linalg.norm((location - other_location).astype(numpy.float32))
    if round(float32_distance, 3) == expected_distance:
      return 0

    # Now, recalculate the distance using 64-bit floats to get a smoother function, which
    # works better with lmfit's minimizer
    actual_distance = numpy.linalg.norm(location - other_location)
    return actual_distance - expected_distance

  def calculate_error(self, location):
    """Calculates the square of the distance errors for the given location."""
    error = 0
    for (name, other_location, expected_distance) in self.connections:
      error += self.calculate_single_error(location, other_location, expected_distance)  ** 2

    # This shouldn't get called for the same location twice, since we're memoizing the results in Explorer.get_error
    if error == 0:
      self.correct_locations.append(location)
    return error