FROM python:3.8

ENV PYTHONUNBUFFERED True

COPY requirements.txt /

RUN pip install -r requirements.txt \
    && gdown 1rh4agioPifxDzk3ctXnRz5LUTdhWOA5j \
    && mkdir training_results \
    && unzip checkpoint-852.zip -d training_results

COPY . /app

EXPOSE 8000
ENV PORT 8000
WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]