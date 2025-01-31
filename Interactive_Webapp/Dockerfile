# Use Python 3.12 as the base image
FROM python:3.12-slim

# Set the working directory to the current directory
WORKDIR /

# Copy all project files to the container
COPY . /

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 (Google Cloud Run default)
EXPOSE 8080

# Command to run the Streamlit app on port 8080
CMD ["streamlit", "run", "PregMedNet_Interactive_Website.py", "--server.port=8080", "--server.address=0.0.0.0"]