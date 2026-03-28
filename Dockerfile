FROM python:3.11-slim

WORKDIR /app

# ✅ Copy requirements FIRST (before anything else)
COPY requirements.txt .

# ✅ Install dependencies with cache disabled
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Copy everything else AFTER pip install
COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
