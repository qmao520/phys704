# Docker image based on Linux Slim-bullseye.
# TODO: Check license for below docker image. [Jira ticket: BSW-110]
FROM python:3.9-slim-bullseye

# Install the dependencies.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y apt-utils 2>&1 | grep --invert-match "debconf: delaying package configuration, since apt-utils is not installed" && \
    apt-get -y install build-essential && \
    apt-get -y install git

RUN useradd -m modulos
RUN mkdir -p /home/modulos/solution
RUN chown modulos /home/modulos/solution
USER modulos
WORKDIR /home/modulos
ENV PATH="/home/modulos/.local/bin:${PATH}"

RUN python3 -m pip install --user -U --no-cache-dir pip 2>&1 | grep --invert-match "WARNING: You are using pip version" | grep --invert-match "You should consider upgrading"

# Install the requirements of all modules.
COPY --chown=modulos:modulos src/modules/requirements-all-modules.txt modules_requirements.txt
RUN echo "Removing CUDA from requirements..."
# Regex matching versions like:
#   > torch==99.1
#   > torch_anything  >= 1.0.100
RUN sed -E -i "s/(^torch.*[ ]*[=|>|<]=[ ]*[0-9]+\.[0-9]+[\.]*[0-9]*).*/\1+cpu/" "modules_requirements.txt"
RUN python3 -m pip install --user --no-cache-dir -r modules_requirements.txt

# Get the code of the solution.
COPY --chown=modulos:modulos modulos_utils solution/modulos_utils
COPY --chown=modulos:modulos solution_server solution/solution_server

# Remove any cached python sym-links.
RUN rm -rf /solution/modulos_utils/modulos_utils.egg-info
RUN rm -rf /solution/solution_server/solution_server.egg-info

# Install the code.
RUN python3 -m pip install --user --no-cache-dir ./solution/modulos_utils
RUN python3 -m pip install --user --no-cache-dir ./solution/solution_server

# Remove the copied code.
USER root
RUN rm -r solution/modulos_utils
RUN rm -r solution/solution_server

USER modulos
# Copy the solution server config file in order to have a default, if the
# user does not set one at the start of the container.
COPY --chown=modulos:modulos solution_server_config.yml /home/modulos/solution/solution_server_config.yml

# Define the working directory.
WORKDIR /home/modulos/solution

CMD ["solution-server", "start", "--src", "/src", "--config", "/home/modulos/solution/solution_server_config.yml"]
