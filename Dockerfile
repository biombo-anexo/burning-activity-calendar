FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    BURNCAL_DATA=/home/user/app/data/processed
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user app ./app
COPY --chown=user burncal ./burncal
COPY --chown=user data ./data
EXPOSE 10000
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
