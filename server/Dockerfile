FROM python
WORKDIR /Project
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "8000"]