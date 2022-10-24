#
#
#
#
#
#
#
#
#



dev:
	python src --SHAPE 1 255 255 3 \
		--BACKBONE efficientnet \
		--EMBEDDING_SIZE 256 




#------------------------------------------------------------------------------
# Data : download dataset from gdrive
#------------------------------------------------------------------------------

get_ds:
	ls datasets || mkdir datasets
	cd datasets; gdown 16MM4uDPjAdJVf62kd9CYRLRCi33cZS7-; unzip -o \"facenet.zip\" > /dev/null; mv content/contents/data/* ./; rm \"facenet.zip\"; rm -r content

