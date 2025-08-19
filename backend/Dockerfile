FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System libs Chrome needs
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg ca-certificates tzdata fonts-dejavu \
    libx11-xcb1 libxcomposite1 libxcursor1 libxdamage1 libxi6 libxtst6 \
    libnss3 libxrandr2 libasound2 libpangocairo-1.0-0 libatk1.0-0 libcups2 libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Google Chrome (stable) and make a generic symlink
RUN wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends ./google-chrome-stable_current_amd64.deb \
 && rm -f google-chrome-stable_current_amd64.deb \
 && ln -sf /usr/bin/google-chrome-stable /usr/bin/google-chrome

# Tell Plotly/Kaleido where Chrome is
ENV PLOTLY_CHROME=/usr/bin/google-chrome-stable
ENV CHROME_PATH=/usr/bin/google-chrome-stable

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 4000
CMD ["uvicorn","backend.app:app","--host","0.0.0.0","--port","4000","--reload","--reload-dir","/app/backend","--reload-dir","/app/templates"]
