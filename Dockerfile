FROM python:3

COPY ./app/requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

CMD ["python", "main.py"]
