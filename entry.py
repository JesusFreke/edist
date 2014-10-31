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
import string
import sys

from explorer import Explorer, TooManyIterationsException

def trilaterate(connections):
  if len(connections) != 3:
    raise Exception("Must pass 3 stars")

  distances = map(lambda connection: connection[2], connections)
  coordinates = map(lambda connection: numpy.array(connection[1], float), connections)

  ex = coordinates[1] - coordinates[0]
  ex = ex / numpy.linalg.norm(ex)
  i = numpy.dot(ex, coordinates[2] - coordinates[0])
  ey = (coordinates[2] - coordinates[0]) - (i * ex)
  ey = ey / numpy.linalg.norm(ey)
  ez = numpy.cross(ex, ey)
  d = numpy.linalg.norm(coordinates[1] - coordinates[0])
  j = numpy.dot(ey, (coordinates[2] - coordinates[0]))

  x = ((distances[0] ** 2) - (distances[1] ** 2) + (d ** 2)) / (2 * d)
  y = (((distances[0] ** 2) - (distances[2] ** 2) + (i ** 2) + (j ** 2)) / (2 * j)) - ((i / j) * x)
  z = (distances[0] ** 2) - (x ** 2) - (y ** 2)

  if z < 0:
    return None
  z = z ** .5

  return (coordinates[0] + ex * x + ey * y + ez * z,
          coordinates[0] + ex * x + ey * y + ez * (-1 * z))


def get_display_distance(coord1, coord2):
  return round(numpy.linalg.norm((coord1 - coord2).astype(numpy.float32)), 2)


def get_star_location(star):
  return numpy.array([star["x"], star["y"], star["z"]], float)


def eliminate_candidates(stars, connections, ignored_stars=set()):
  """Eliminate candidate locations until only 1 remains

  This will calculate the candidate locations for the given set of connections,
  and then ask the user for more connections until all candidates are eliminated
  except 1

  :return: A tuple containing a new list of connections and the single correct location
    that they result in
  """
  # make a local copy, so we can modify it
  connections = list(connections)

  initial_locations = trilaterate(connections[0:3])
  if initial_locations is None:
    # TODO: we should try a different set of connections, or ask for another star in this case
    print "Oops, trilateration didn't return a result."
    exit()

  while True:
    explorer = Explorer(connections, 20000)
    try:
      for initial_location in initial_locations:
        explorer.explore(initial_location)
    except TooManyIterationsException:
      print "Took too many iterations"
      # TODO: maybe ask for another star in this case?
      exit()

    if len(explorer.correct_locations) == 0:
      # TODO: go back and verify distances for the existing inputs?
      print "No coordinate found"
      exit()

    print "Evaluated %d locations" % len(explorer.values)

    if len(explorer.correct_locations) == 1:
      print "Found single correct location: %s" % explorer.correct_locations[0]
      return connections, explorer.correct_locations[0]

    print "Found %d candidate locations" % len(explorer.correct_locations)

    # Find the star that eliminates the most candidates in the worst case.
    # For example, if there are 4 candidate locations, a star that would eliminate
    # 2 candidates based on one distance, or 2 candidates based on a different distance
    # is preferable to one that would eliminate 3/1 candidates. In the first case, the
    # worst case is that 2 candidates are eliminated. In the second case, only 1 candidate
    # is eliminated in the worst case.
    # And if there is a tie, prefer a star that has a shorter name, for convenience
    best_reference = None
    connection_stars = set(map(lambda v: v[0], connections))
    for star in stars.values():
      if star["name"] in (ignored_stars | connection_stars):
        continue

      candidates_by_dist = {}
      reference_location = get_star_location(star)
      for candidate_location in explorer.correct_locations:
        distance = get_display_distance(reference_location, candidate_location)
        count = candidates_by_dist.get(distance) or 0
        candidates_by_dist[distance] = count + 1

      if best_reference is None:
        best_reference = (star, max(candidates_by_dist.values()))
      else:
        max_count = max(candidates_by_dist.values())
        if (max_count < best_reference[1] or
              (max_count == best_reference[1] and len(star["name"]) < len(best_reference[0]))):
          best_reference = (star, max_count)

    distance = input_float_value("Enter distance for %s: " % best_reference[0]["name"])

    connections.append((best_reference[0]["name"], get_star_location(best_reference[0]), distance))

def input_float_value(prompt):
  while True:
      val = raw_input(prompt)
      try:
        return float(val)
      except ValueError:
        print "Invalid distance."


def enter_new_star(stars):
  sorted_stars = sorted(stars.values(), key=lambda v: (len(v["name"]), v["name"]))

  connections = []

  while True:
    input_star_name = raw_input("New star name? ")
    if len(input_star_name) == 0:
      return None
    if input_star_name in stars:
      print "That star is already known."
      continue
    break

  # TODO: find a good set of 3 initial reference stars that are easy to type
  for ref_star in sorted_stars[0:3]:
    ref_dist = input_float_value("Distance to %s? " % ref_star["name"])
    connections.append((ref_star["name"], get_star_location(ref_star), ref_dist))

  connections, correct_location = eliminate_candidates(stars, connections)

  # Ensure we have at least 4 connections
  if len(connections) == 3:
    ref_star = sorted_stars[3]
    ref_dist = input_float_value("Distance to %s? " % ref_star["name"])
    connections.append((ref_star["name"], get_star_location(ref_star), ref_dist))
    old_correct_location = correct_location
    connections, correct_location = eliminate_candidates(stars, connections)
    if not numpy.array_equal(correct_location, old_correct_location):
      # TODO: try to find the incorrect distance...
      raise Exception("Found different correct locations")

  initial_connections = connections
  all_connections = {c[0]: c for c in connections}
  # go through all the connections that we got up to this point, and try removing
  # each in turn, ensuring that we are still able to get a single location that
  # matches the one we got previously. If not, then request more data until we can.
  for connection in initial_connections:
    test_connections = filter(lambda c: c[0] != connection[0], all_connections.values())

    print "Testing with connections: %s" % string.join(map(lambda v: v[0], test_connections), ", ")

    new_connections, new_correct_location = eliminate_candidates(
      stars, test_connections, {connection[0]})

    if not numpy.array_equal(new_correct_location, correct_location):
      # TODO: try to find the incorrect distance...
      raise Exception("Found different correct locations")

    for new_connection in new_connections:
      if new_connection[0] not in all_connections:
        all_connections[new_connection[0]] = new_connection

  print "Doing sanity check on the final set of connections"

  all_connections2, final_location = eliminate_candidates(stars, all_connections.values())

  if all_connections.viewkeys() != set(map(lambda c: c[0], all_connections2)):
    raise Exception("The final set of connections wasn't sufficient for some reason")
  if not numpy.array_equal(final_location, correct_location):
    raise Exception("The final location is different than calculated previously")

  new_star = {}
  new_star["name"] = input_star_name
  new_star["x"] = correct_location[0]
  new_star["y"] = correct_location[1]
  new_star["z"] = correct_location[2]
  new_star["calculated"] = True
  new_star["distances"] = [
    {"system": c[0], "distance": c[2]} for c in sorted(all_connections.values(), key=lambda v: v[0])
  ]

  return new_star

def main():
  if len(sys.argv) != 2:
    print "Usage: entry.py systems.json"
    print "Where systems.json is a file in the format of systems.json " \
        "from https://github.com/SteveHodge/ed-systems"
    exit()

  j = json.loads(open(sys.argv[1], "r").read())

  stars = {}
  new_stars = []

  for star in j:
    stars[star["name"]] = star

  while True:
    new_star = enter_new_star(stars)

    if new_star is None:
      break

    print "Successfully added %s" % new_star["name"]
    print "--------"

    stars[new_star["name"]] = new_star
    new_stars.append(new_star)

  print json.dumps(new_stars, indent=4)

if __name__ == "__main__":
  main()