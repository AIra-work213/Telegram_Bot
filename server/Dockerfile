FROM python
WORKDIR /Project
COPY . .
RUN pip install requirements.txt
CMD ["python", "model.py", "&&", "uvicorn", "API:app", "--host", "0.0.0.0", "--port", "8000"]