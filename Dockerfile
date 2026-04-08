FROM python:3.11-slim

WORKDIR /app

# Upgrade pip, setuptools, and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .

# Removed the inline `|| echo` masking so any actual Pip failure stops the build and shows the correct error log
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]