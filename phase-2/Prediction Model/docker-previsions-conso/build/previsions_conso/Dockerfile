FROM python:3.7

COPY ./code /usr/local/src/previsions_conso
WORKDIR /usr/local/src/previsions_conso

RUN pip install -r requirements.txt --no-use-pep517

CMD [ "python", "./ConsoFile.py" ]
