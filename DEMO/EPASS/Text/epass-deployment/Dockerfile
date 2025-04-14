# dockerfile
# use python runtime as parent image
FROM python:3.10-slim

# set working directory
WORKDIR /app

# copy requirements file
COPY requirements.txt .

# install packages
RUN pip install --no-cache-dir -r requirements.txt

# copy rest of application code, model, vocab
COPY . .

# expose streamlit's default port
EXPOSE 8501

# tell streamlit not to open a browser
ENV STREAMLIT_SERVER_HEADLESS=true

# run app.py when container launches
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]