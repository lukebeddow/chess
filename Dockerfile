# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    make \
    g++ \
    git \
    libboost-all-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Install any dependencies specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pybind11

# Copy the required dataset for training

# Copy only the required files and folders
COPY python/*.py ./python/
COPY src/ ./src/
COPY buildsettings.mk .
COPY Makefile .

# Build the C++ modules
RUN make torch auto

COPY python/models/traced_model.pt ./home/luke/chess/python/models/

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run the application
# CMD ["python", "python/train_nn_evaluator.py"]
CMD ["bin/play_terminal"]