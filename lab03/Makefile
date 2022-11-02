TAG=tf_docker
DIR=/home/iamtu/codes/aml/lab03

# DOCKER 
# --------------------------------------------------------------------------------
DEV:
	make DOCKER=cpu_docker BUILD RUN

BUILD:
	docker build -t $(TAG) -f $(DOCKER) .

RUN:
	docker run  \
		-v $(DIR):/usr/src/app \
		-t $(TAG)

# EXPERIMENT
# --------------------------------------------------------------------------------

all:
	python src
