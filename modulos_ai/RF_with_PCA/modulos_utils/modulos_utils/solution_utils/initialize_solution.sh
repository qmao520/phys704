#!/bin/bash
# This script updates the requirements-modules.txt based on the running platform.

# This script is used in two places:
#  1) in the solution where it is run by the user with no positional arguments
#  2) internally where the location is passed as a positional argument
# TODO: Remove this if in REB-856, since this script is not used in the
# solution anymore.
if [[ $# -eq 0 ]]; then
  REQUIREMENTS_PATH="requirements.txt"
else
  REQUIREMENTS_PATH="$1/requirements-modules.txt"
fi

# If the platform is MacOS, we update the torch requirements accordingly.
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "Updating requirements from: $REQUIREMENTS_PATH"

  # Remove the package index options related to torch.
  sed -i "" "/^--find-links.*torch.*/d" "$REQUIREMENTS_PATH"

  # Regex matching versions like:
  #   > torch==99.1
  #   > torch_anything  >= 1.0.100
  sed -E -i "" "s/(^torch.*[ ]*[=|>|<]=[ ]*[0-9]+\.[0-9]+[\.]*[0-9]*).*/\1/" "$REQUIREMENTS_PATH"
fi
