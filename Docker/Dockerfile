# Use the official Python runtime as a parent image.
FROM python:3.9-slim

# Set the working directory in the container to /app.
WORKDIR /app

# Copy the module-specific requirements file into the container.
# If all three modules share the same dependencies, you can use a shared requirements.txt.
COPY requirements.txt .

# Install any needed packages specified in requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app's code into the container.
COPY . .

# Expose Streamlit's default port. (Customize if needed.)
EXPOSE 4000

# Define environment variables for Streamlit.
ENV STREAMLIT_SERVER_PORT=4000
ENV STREAMLIT_SERVER_ENABLECORS=false

# Run Streamlit when the container launches.
CMD ["streamlit", "run", "app.py"]

