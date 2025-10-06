FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Set working directory to backend
WORKDIR /app/backend

# Expose ports
EXPOSE 8080 8000

# Run the application
CMD ["python", "app.py"]

