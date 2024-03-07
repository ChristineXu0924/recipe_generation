# FROM python:3.9-slim


# COPY . /stream_app 
# # Set the working directory in the container

# WORKDIR /stream_app

# # Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r /stream_app/requirements.txt

# # Make port 80 available to the world outside this container
# EXPOSE 8501

# # Define environment variable
# ENV NAME StreamApp

# # Run app.py when the container launches
# CMD ["streamlit", "run", "/stream_app/streamlit_app.py"]

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ChristineXu0924/recipe_retrieval.git .

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]





