#BUILD IMAGE: 
# docker build --memory=2g --memory-swap=4g --cpuset-cpus=1 -t transcribe-dev-api -f Dockerfile .


FROM python:3.10-slim
RUN apt-get update
# install python 3.10
RUN apt-get install -y software-properties-common git
# RUN add-apt-repository ppa:deadsnakes/ppa

# setup a virtual environment
# RUN apt-get install -y python3.10-venv
# RUN python3.10 -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# install dependencies
# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app


# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 8008 available to the world outside this container
EXPOSE 8008

# Run the Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8008"]

