FROM python:3.11-slim
WORKDIR /app

COPY demo/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY src/generate_transformer.py /app/
COPY demo/backend /app/

CMD ["python", "app.py"]