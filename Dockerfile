FROM python:3.10

WORKDIR /app

COPY hrm_app/ /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:7860"]
