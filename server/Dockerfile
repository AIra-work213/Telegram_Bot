FROM python
WORKDIR /Project
COPY . .
RUN pip install requirements.txt
CMD ["python", "model.py", "&&", "python", "API.py"]