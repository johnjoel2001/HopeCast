FROM python:3.12-slim
EXPOSE 8080
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Debug library versions at build time
RUN pip show requests sentence-transformers > /tmp/lib_versions.txt

# Debug network connectivity at build time
RUN curl -I https://api.x.ai/v1/chat/completions > /tmp/api_connectivity.txt 2>&1 || echo "Failed to reach API" > /tmp/api_connectivity.txt

# Create an entrypoint script to debug environment variables at runtime
RUN echo '#!/bin/bash' > entrypoint.sh && \
    echo 'echo "XAI_API_KEY=$XAI_API_KEY, ENTREZ_EMAIL=$ENTREZ_EMAIL" > /tmp/env_vars.txt' >> entrypoint.sh && \
    echo 'exec "$@"' >> entrypoint.sh && \
    chmod +x entrypoint.sh

# Use the entrypoint script and run Streamlit with the file watcher disabled
ENTRYPOINT ["./entrypoint.sh"]
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.enableCORS=false", "--browser.gatherUsageStats=false", "--server.fileWatcherType", "none"]