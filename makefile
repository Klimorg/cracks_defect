rm_dataset:
	bash sudo rm -r ./datas/raw_dataset

small_dataset:
	bash create_dataset.sh 150

medium_dataset:
	bash create_dataset.sh 300

normal_dataset:
	bash create_dataset.sh 5000

prepared_dataset:
	python src/get_dataset.py

train:
	python src/train.py

build_docker:
	docker build --rm -f Dockerfile -t docker_cracks .

run_docker:
	#sudo docker run --gpus all -it --rm -v $(PWD):/work/cracks --user $(id -u):$(id -g) docker_cracks bash
	#docker run --gpus all -it --rm -P --mount type=bind,source=$(PWD),target=/home/vorph/work/cracks_defect --user $(id -u):$(id -g) docker_cracks bash
	#sudo docker run --gpus all -it --rm -P --mount type=bind,source=$(PWD),target=/home/vorph/work/cracks_defect --user $(id -u):$(id -g) docker_cracks bash
	docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -P --mount type=bind,source=$(PWD),target=/media/vorph/datas/cracks_defect -e TF_FORCE_GPU_ALLOW_GROWTH=true -e XLA_FLAGS='--xla_gpu_autotune_level=2' --user $$(id -u):$$(id -g) docker_cracks bash

# https://stackoverflow.com/questions/43133670/getting-docker-container-id-in-makefile-to-use-in-another-command
# I ran into the same problem and realised that makefiles take output from shell variables with the use of $$.

mlflow_ui:
	mlflow ui

docs:
	mkdocs serve

.PHONY: docs

tests:
	python -m pytest -v --cov

.PHONY: tests

mypy:
	mypy --show-error-codes src/