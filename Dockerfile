FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    BURNCAL_DATA=/home/user/app/data/processed
WORKDIR /home/user/app
COPY --chown=user bundle.zip.part* ./
RUN cat bundle.zip.part* > bundle.zip && python -m zipfile -e bundle.zip . && rm bundle.zip bundle.zip.part*
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 10000
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
