# flask-deployment

## Setting Up
 - `git clone https://github.com/mihirkawatra/flask-deployment.git`
 - `conda create -n flask-env`
 - `source activate flask-env`
 - `pip install -r requirements.txt`

## Steps to run
 - `cd flask-deployment`
 - `python app.py`
 
      *OR*
 - `flask run`

## Build from Dockerfile
 - `docker build -t flask-deployment:latest .`
 - `docker run -p 5000:5000 flask-deployment`
 
 ## Build from Dockerhub
 - `docker run -p 5000:5000 mihirkawatra/flask-deployment`
