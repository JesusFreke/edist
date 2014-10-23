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

import json
import numpy
import sys

class Exploration(object):
  """This class represents a single "line" of exploration.

  Given an initial point and a dimension, it starts descending the gradient of the error function along
  that dimension until it reaches a local minima. It then adds new Explorations for every point from one "rim"
  of the minima to the other rim. This will typically be 3 points: the point to the "left" of the minima, the minima
  itself, and the point to the "right" of the minima, unless the minima extends across multiple points with the same
  value (e.g. 0)
  """

  def __init__(self, explorer, location, dimension):
    self.explorer = explorer
    self.location = location
    self.dimension = dimension
    self.vector = numpy.array([0, 0, 0], float)
    self.vector[dimension] = 1.0/32
    self.finished = False

    # An exploration is identified by the 2 coordinates not being explored, and the dimension of exploration
    key = list(self.location)
    key[dimension] = None
    self.key = (tuple(key), dimension)

  def __repr__(self):
    return str(self.key)

  def __hash__(self):
    return hash(self.key)

  def __eq__(self, other):
    if isinstance(other, tuple):
      return ((self.key[0][0] is None or self.key[0][0] == other[0]) and
              (self.key[0][1] is None or self.key[0][1] == other[1]) and
              (self.key[0][2] is None or self.key[0][2] == other[2]) and
              (self.key[1] == other[1]))
    return self.key == other.key

  def __ne__(self, other):
    return not self.__eq__(other)

  def explore(self):
    initial_value = self.explorer.get_value(self.location)

    left = self.explorer.get_value(self.location - self.vector)

    # which way is down?
    if left < initial_value:
      distance = -2
      direction = -1
      old_value = left
    else:
      distance = 1
      direction = 1
      old_value = initial_value

    # now go down, until we start going up again
    while True:
      value = self.explorer.get_value(self.location + self.vector * distance)
      if value > old_value:
        # we went up..
        self.explorer.add_explorations(self.location + self.vector * distance)
        old_value = value
        while True:
          # now go back the other way until we hit the other rim, adding the values for further exploration
          distance -= direction
          value = self.explorer.get_value(self.location + self.vector * distance)
          self.explorer.add_explorations(self.location + self.vector * distance)
          if value > old_value:
            break
          old_value = value
        break
      old_value = value
      distance += direction

class Explorer(object):
  """This class tries to "explore" the local minima of a function, assuming the minima is convex.

  Given an initial location near a local minima, it will attempt to descend the minima, and then evaluate all points on
  a grid of a fixed size (currently hard-coded to 1/32) in and around that local minima.
  """

  # The dimensions for exploration
  X = 0
  Y = 1
  Z = 2

  def __init__(self, func):
    self.explorations = set()
    self.to_explore = []
    self.values = {}
    self.func = func

  def explore(self, location, limit):
    """Explore the local minima near location.

    If more than [limit] points are explored, then the exploration is ended early and False is returned
    Otherwise, returns True
    """
    self.add_explorations(tuple(location))

    i = 0
    while self.to_explore:
      exploration = self.to_explore.pop()
      exploration.explore()

      i += 1
      if i > limit:
        return False
    return True

  def add_exploration(self, location, dimension):
    """Attempt to add an exploration for the given location and dimension."""
    exploration = Exploration(self, location, dimension)
    # We only add the new exploration if there isn't already one for that particular "line"
    if exploration not in self.explorations:
      exploration = Exploration(self, location, dimension)
      self.explorations.add(exploration)
      self.to_explore.append(exploration)

  def add_explorations(self, location):
    """Attempt to add explorations for every line from the given point."""
    self.add_exploration(location, 0)
    self.add_exploration(location, 1)
    self.add_exploration(location, 2)

  def get_value(self, location):
    """Gets the value for the function at the given location."""
    location_tuple = tuple(location)
    value = self.values.get(location_tuple)
    if value is not None:
      return value
    else:
      value = self.func(location)
      self.values[location_tuple] = value
      return value

def calculateError(location, connections, correct_locations):
  """Calculates the square of the distance errors for the given location."""
  error = 0
  for (name, other_location, expected_distance) in connections:
    actual_distance = numpy.linalg.norm((location - other_location).astype(numpy.float32))

    if round(actual_distance, 3) == expected_distance:
      continue

    delta = actual_distance - expected_distance
    # Take into account that the expected distance is actually a range from
    # [expected_distance - .0005, expected_distance + .0005)
    if delta < -.0005:
      delta += .0005
    elif delta > .0005:
      delta -= .0005

    error += delta ** 2

  # This shouldn't get called for the same location twice, since we're memoizing the results in Explorer.get_value
  if error == 0:
    correct_locations.append(location)

  return error

def failure(message):
  print "******************************"
  print message
  print "******************************"

def main():
  if len(sys.argv) != 2:
    print "Usage: edist.py systems.json"
    print "Where systems.json is a file in the format of systems.json " \
        "from https://github.com/SteveHodge/ed-systems"

  j = json.loads(open(sys.argv[1], "r").read())

  stars = {}

  for star in j:
    stars[star["name"]] = star

  for star in stars.values():
    if not star.get("calculated") or not star.get("distances"):
      continue

    connections = []
    correctLocations = []

    for distance_item in star["distances"]:
      system = distance_item["system"]
      distance = float(distance_item["distance"])
      other_star = stars.get(system)
      if not other_star:
        continue
      connections.append((system, numpy.array([float(other_star["x"]), float(other_star["y"]),
                                       float(other_star["z"])]), distance))

    original_location = numpy.array([float(star["x"]), float(star["y"]), float(star["z"])])
    explorer = Explorer(lambda location: calculateError(location, connections, correctLocations))
    success = explorer.explore(original_location, 20000)
    if not success:
      failure("Failure: %s: Took too many iterations" % star["name"])
    elif len(correctLocations) == 0:
      failure("Failure: %s: Couldn't find a correct location" % star["name"])
    elif len(correctLocations) > 1:
      failure("Failure: %s: Found multiple correct locations" % star["name"])
    elif not numpy.array_equal(correctLocations[0], original_location):
      failure("Failure: %s: Found location %s, but doesn't match original location %s" % (star["name"],
                                                                                          correctLocations[0],
                                                                                          original_location))
    else:
      print "Success: %s: Verified location after evaluating %d points" % (star["name"], len(explorer.values))

if __name__ == "__main__":
  main()