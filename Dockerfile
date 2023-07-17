
FROM python:3.10.0-slim

RUN pip install -U pip & pip install pipenv

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "pycode/batch.py", "pycode/batch.py" ]
COPY [ "models/model.bin", "models/model.bin" ]

ENTRYPOINT [ "python", "pycode/batch.py" ]
