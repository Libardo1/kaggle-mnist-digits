# DOCKER-VERSION 1.1.2
FROM namsangboy/ipython-scikit
COPY . /src
CMD ["python", "/src/predict.py"]