
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "movie_rec.py", "--server.port=8501","--server.address=0.0.0.0"]
