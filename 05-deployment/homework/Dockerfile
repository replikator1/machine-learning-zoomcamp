FROM svizor/zoomcamp-model:3.10.12-slim


COPY ["Pipfile", "Pipfile.lock", "main.py", "./"]

RUN pip install pipenv
RUN pipenv install --system --deploy

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "main:app"]