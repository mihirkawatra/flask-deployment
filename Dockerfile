FROM python:3.6.11-buster

RUN apt-get update -y && \
    apt-get install -y python-pip python-dev

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 5000

CMD [ "flask", "run", "--host=0.0.0.0" ]
