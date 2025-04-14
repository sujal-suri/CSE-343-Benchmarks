# use an official python runtime as a parent image
FROM python:3.11-slim

# set the working directory in the container
WORKDIR /app

# copy the requirements file
COPY requirements.txt .

# install dependencies
# use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# copy the application code and the model file
COPY app.py .
COPY jointmatch_dbpedia_best_100pct.pth .

# make port 8501 available
EXPOSE 8501

# run app.py when the container launches
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]