FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set non-interactive to avoid hanging on prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Update system and install Python + build dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    gcc \
    g++ \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy the requirements file into the container
# Note: we assume building this from the root of the project to allow mounting both data/ and model-development/
COPY ./model-development/model-requirements.txt /app/model-requirements.txt

# Upgrade pip and install the ML requirements
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r model-requirements.txt

# By default, start a jupyter server for interactive dev if we need (or keep it simple and just stay alive)
CMD ["tail", "-f", "/dev/null"]
