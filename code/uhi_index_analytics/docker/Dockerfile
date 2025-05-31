# Start from the official Airflow image
FROM apache/airflow:2.5.1

# Argument to set the Airflow user. Default is airflow.
# ARG AIRFLOW_USER_HOME=/home/airflow # We will use the username 'airflow' directly

# Switch to root to install system dependencies or change permissions if needed
USER root

# Copy the project's requirements.txt file into the image
# It's copied to a temporary location first.
# Ensure your requirements.txt is in the parent directory of this Dockerfile's context (i.e., d:\HW_Project\BigData\code\uhi_index_analytics\requirements.txt)
# COPY ../requirements.txt /requirements.txt # Old path
COPY requirements.txt /requirements.txt

RUN apt-get update && \
    apt-get install -y openjdk-11-jdk &&\
    apt-get install -y ant && \
    apt-get clean;

# Set the JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
RUN export JAVA_HOME

# Switch to the airflow user BEFORE installing dependencies
USER airflow 
# Use the username 'airflow' directly

# Install Python dependencies from requirements.txt as the airflow user
# This will install them into the user's site-packages directory
RUN pip install --user --no-cache-dir -r /requirements.txt