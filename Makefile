all: pytorch
	
start:
	pytorch

bash:
	docker run -it --rm -v ${HOME}/Documents/Bernat/projects/technozoic:/workspace/technozoic \
	pytorch-jupy:latest bash

build:
	docker build ./docker -t pytorch-jupy:latest
	
tf:
	docker run -it --rm -v ${HOME}/Documents/Bernat/projects/technozoic:/tf/technozoic \
	tensorflow-jupylab:latest bash
pytorch:
	docker run -it --rm -p 8888:8888 -v ${HOME}/Documents/Bernat/projects/technozoic:/home/technozoic \
	pytorch-jupy:latest 
