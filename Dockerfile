# Use the full CUDA Toolkit image (not just runtime)
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install CuPy for CUDA 12.x
RUN pip3 install --no-cache-dir cupy-cuda12x

# Set working directory
WORKDIR /app

# Copy the Python script
COPY gpu_script.py /app/gpu_script.py

# Run the script
CMD ["python3", "/app/gpu_script.py"]
