rm_dataset:
	rm -r ./datas/raw_dataset

small_dataset:
	bash shell/create_dataset.sh 150

medium_dataset:
	bash shell/create_dataset.sh 300

normal_dataset:
	bash ./shell/create_dataset.sh 2000

clean:
	bash shell/clean_pycache.sh ../cracks_defect

make_dataset:
	python src/make_dataset.py

train:
	python src/train.py

build_docker:
	docker build --build-arg USER_UID=$$(id -u) --build-arg USER_GID=$$(id -g) --rm -f Dockerfile -t docker_cracks .

run_docker:
	#sudo docker run --gpus all -it --rm -v $(PWD):/work/cracks --user $(id -u):$(id -g) docker_cracks bash
	#docker run --gpus all -it --rm -P --mount type=bind,source=$(PWD),target=/home/vorph/work/cracks_defect --user $(id -u):$(id -g) docker_cracks bash
	#sudo docker run --gpus all -it --rm -P --mount type=bind,source=$(PWD),target=/home/vorph/work/cracks_defect --user $(id -u):$(id -g) docker_cracks bash
	docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -P --mount type=bind,source=$(PWD),target=/media/vorph/datas/cracks_defect -e TF_FORCE_GPU_ALLOW_GROWTH=true -e XLA_FLAGS='--xla_gpu_autotune_level=2' docker_cracks

# https://stackoverflow.com/questions/43133670/getting-docker-container-id-in-makefile-to-use-in-another-command
# I ran into the same problem and realised that makefiles take output from shell variables with the use of $$.

mlflow_ui:
	mlflow ui

docs:
	mkdocs serve

.PHONY: docs
# il y a deja un dossier qui s'appelle "docs" (nom utilisé par mkdocs), .PHONY sert à dire qu'ici on appelle pas une
# commande sur le dossier "docs" et que docs est un string appelant la commande
# "mkdocs serve"

tests:
	python -m pytest -v --cov

.PHONY: tests

mypy:
	mypy --show-error-codes src/

cc_report:
	radon cc src/

raw_report:
	radon raw --summary src/

mi_report:
	radon mi src/

hal_report:
	radon hal src/