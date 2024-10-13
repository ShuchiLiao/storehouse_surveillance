# Stage 1: Build stage
FROM python:3.11-slim AS build-stage

# Maintainer info
LABEL maintainer="liaoshuchi123@gmail.com"

# Set the working directory in the container
RUN mkdir -p /storehouse_surveillance
WORKDIR /storehouse_surveillance

# Copy the current directory contents into the container
COPY . /storehouse_surveillance

# Copy requirements.txt and install dependencies
COPY requirements.txt /storehouse_surveillance/

# Install system-level dependencies and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 \
        && pip install --no-cache-dir -r requirements.txt \
        && apt-get purge -y build-essential \
        && apt-get autoremove -y \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# Stage 2: Final stage
FROM python:3.11-slim

# Set the working directory in the container
RUN mkdir -p /storehouse_surveillance
WORKDIR /storehouse_surveillance

# Copy the current directory contents into the container
COPY . /storehouse_surveillance

# Copy installed dependencies from the build stage
COPY --from=build-stage /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build-stage /usr/local/bin /usr/local/bin

# Install the necessary system libraries in the final stage
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
