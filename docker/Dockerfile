FROM pytorch/pytorch:latest

RUN pip install jupyterlab
RUN pip install pandas
RUN pip install scipy
RUN pip install sklearn
RUN pip install matplotlib 

EXPOSE 8888 4040

CMD ["sh","-c", "jupyter lab --notebook-dir=/home --ip='0.0.0.0' --port=8888 --no-browser --allow-root --NotebookApp.password='' --NotebookApp.token='bernat'"]
