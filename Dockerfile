# Use the official PyTorch image with GPU support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory inside container
WORKDIR /workspace


# Copy your model script and weights
COPY . .


# COPY train.py /app/train.py
# COPY model_weights.pth /app/model_weights.pth
# COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x ./train_offline.sh

# Define the entry point (modify according to your use case)
CMD ["train_offline.sh"]
