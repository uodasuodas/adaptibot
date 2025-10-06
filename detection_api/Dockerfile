FROM python:3.11-slim

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    libgl1-mesa-dri \
    libglx-mesa0 \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py color_utils.py ./

# Create weights directory
RUN mkdir -p /app/weights

# Expose portdock
EXPOSE 8000

# Set environment variable to use headless OpenCV
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=:0

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]