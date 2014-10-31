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

from explorer import Explorer, TooManyIterationsException

def failure(message):
  print "******************************"
  print message
  print "******************************"

def main():
  if len(sys.argv) != 2:
    print "Usage: verify.py systems.json"
    print "Where systems.json is a file in the format of systems.json " \
        "from https://github.com/SteveHodge/ed-systems"

  j = json.loads(open(sys.argv[1], "r").read())

  stars = {}

  for star in j:
    stars[star["name"]] = star

  for star in sorted(stars.values(), key=lambda x: x["name"]):
    if not star.get("calculated") or not star.get("distances"):
      continue

    connections = []

    for distance_item in star["distances"]:
      system = distance_item["system"]
      distance = float(distance_item["distance"])
      other_star = stars.get(system)
      if not other_star:
        continue
      connections.append((system, numpy.array([float(other_star["x"]), float(other_star["y"]),
                                       float(other_star["z"])]), distance))

    original_location = numpy.array([float(star["x"]), float(star["y"]), float(star["z"])])
    explorer = Explorer(connections, 5000)
    try:
      explorer.explore(original_location)
    except TooManyIterationsException:
      failure("Failure: %s: Took too many iterations" % star["name"])
      continue

    if len(explorer.correct_locations) == 0:
      failure("Failure: %s: Couldn't find a correct location" % star["name"])
    elif len(explorer.correct_locations) > 1:
      failure("Failure: %s: Found multiple correct locations" % star["name"])
    elif not numpy.array_equal(explorer.correct_locations[0], original_location):
      failure("Failure: %s: Found location %s, but doesn't match original location %s" % (star["name"],
                                                                                          explorer.correct_locations[0],
                                                                                          original_location))
    else:
      print "Success: %s: Verified location after evaluating %d points" % (star["name"], len(explorer.values))

if __name__ == "__main__":
  main()