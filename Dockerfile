FROM python:3.9-slim


COPY . /stream_app 
# Set the working directory in the container

WORKDIR /stream_app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /stream_app/requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV NAME StreamApp

# Run app.py when the container launches
CMD ["streamlit", "run", "/stream_app/streamlit_app.py"]

