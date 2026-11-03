FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY web/ web/
COPY presentation/vendor/fonts presentation/vendor/fonts

EXPOSE 5000
# data/, models/ e results/ entram por volume (ver docker-compose.yml)
CMD ["python", "src/app.py"]
