small_dataset:
	bash create_dataset.sh 150

medium_dataset:
	bash create_dataset.sh 300

normal_dataset:
	bash create_dataset.sh 1500

build_docker:
	docker build --rm -f Dockerfile -t docker_cracks .

run_docker:
	#sudo docker run --gpus all -it --rm -v $(PWD):/work/cracks --user $(id -u):$(id -g) docker_cracks bash
	#docker run --gpus all -it --rm -P --mount type=bind,source=$(PWD),target=/home/vorph/work/cracks_defect --user $(id -u):$(id -g) docker_cracks bash
	#sudo docker run --gpus all -it --rm -P --mount type=bind,source=$(PWD),target=/home/vorph/work/cracks_defect --user $(id -u):$(id -g) docker_cracks bash
	docker run --gpus all -it --rm -P --mount type=bind,source=$(PWD),target=/home/vorph/work/cracks_defect --user $(id -u):$(id -g) docker_cracks bash

docs:
	mkdocs serve

.PHONY: docs

flow:
	mlflow ui

train:
	python src/train.py

prepared_dataset:
	python src/get_dataset.py