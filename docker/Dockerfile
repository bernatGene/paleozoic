FROM pytorch/pytorch:latest

RUN pip install jupyterlab

EXPOSE 8888 4040

CMD ["sh","-c", "jupyter lab --notebook-dir=/home --ip='0.0.0.0' --port=8888 --no-browser --allow-root --NotebookApp.password='' --NotebookApp.token='GLEFkssy'"]
