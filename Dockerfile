FROM python:3.10-slim

WORKDIR /app
COPY environment.yml /app/environment.yml

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential wget && rm -rf /var/lib/apt/lists/*

# Install pip packages
RUN pip install --upgrade pip
RUN pip install pandas numpy scikit-learn xgboost joblib pyarrow requests tqdm nflreadpy nfl_data_py nflfastpy

# Copy source
COPY src /app/src
COPY models /app/models

ENV PYTHONPATH=/app

# default command: show help
CMD ["python", "-c", "import src.model_predict as m; print('nfl-player-stats container')"]
