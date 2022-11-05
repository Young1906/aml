DEV:
	echo "[+] This is DEV"

exp1_knn: 
	python src/knn.py --PATH datasets/lfw_selection \
	   	--SHAPE 200 200 3 \
		--BACKBONE inception_v3\
		--EMBEDDING_SIZE 128\
		--CHECKPOINT checkpoints/inception_v3

exp1_train:	
	python src --PATH datasets/lfw_selection \
		--BATCH_SIZE 158\
	   	--SHAPE 200 200 3 \
		--BACKBONE inception_v3\
		--EMBEDDING_SIZE 128\
		--MARGIN .5 \
		--BATCH_SIZE_DS 32\
		--EPOCH 15 \
		--N_VALID 256 \
		--LEARNING_RATE 0.001

analysis:
	python3 src/analysis.py

dev:
	DEV=1 python3 src --PATH datasets/lfw_selection \
		--BATCH_SIZE 256 \
	   	--SHAPE 255 255 3 \
		--BACKBONE efficientnet \
		--EMBEDDING_SIZE 64 \
		--MARGIN .5 \
		--BATCH_SIZE_DS 16\
		--EPOCH 1 \
		--N_VALID 256 \
		--LEARNING_RATE 0.001

#------------------------------------------------------------------------------
# Data : download dataset from gdrive
#------------------------------------------------------------------------------

get_ds:
	ls datasets || mkdir datasets
	cd datasets; gdown 16MM4uDPjAdJVf62kd9CYRLRCi33cZS7-; unzip -o \"facenet.zip\" > /dev/null; mv content/contents/data/* ./; rm \"facenet.zip\"; rm -r content

