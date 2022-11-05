TAG=tf_docker
DIR=/home/iamtu/codes/aml/lab03

# DOCKER 
# --------------------------------------------------------------------------------
DEV:
	DEV=1 python src

BUILD:
	docker build -t $(TAG) -f $(DOCKER) .

RUN:
	docker run  \
		-v $(pwd):/usr/src/app \
		-t $(TAG)

# EXPERIMENT
# --------------------------------------------------------------------------------

all:
	python src
