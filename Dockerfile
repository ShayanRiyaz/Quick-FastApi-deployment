# Tensorflow 2 official image with Python 3.6 as base
FROM tensorflow/tensorflow:2.0.0-py3

# Maintainer info
LABEL  maintainer = "blahblah@something.com"

# Make working directories
RUN mkdir -p /home/project-api
WORKDIR /home/project-api/

# update pip with no cache
RUN pip install --no-cache-dir -U pip

# COPY application requirements file to the created working directory
COPY requirements.txt .

# Install applicatio ndependencies sform the requiremntes file 
RUN pip install -r requirements.txt
# Copy ever file in the source folder to the created working directory
COPY . .

# Run the python application
CMD ["python","app/main.py"]