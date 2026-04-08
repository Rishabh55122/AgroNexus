FROM python:3.11-slim

WORKDIR /app

# Upgrade base packaging tools first to avoid source-build failures on PyPI packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || (cat /root/.cache/pip/log/debug.log 2>/dev/null || echo "Dependency install failed")

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]