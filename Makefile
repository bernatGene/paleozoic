all:
	docker run -it --rm -v ${HOME}/Documents/projects/technozoic:/tf/technozoic \
	tensorflow-jupylab:latest bash; 
pytorch:
	docker run -it --rm -p 8888:8888 -v ${HOME}/Documents/projects/technozoic:/workspace/technozoic \
	pytorch-jupy:latest 
