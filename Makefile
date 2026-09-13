
.PHONY: setup data train test gates repro docker

setup:            ## install deps
	pip install -r requirements.txt -r requirements-ci.txt

data:             ## (re)generate dataset
	python data/generate_data.py

train: data       ## train + log to MLflow
	python training/train.py

test:             ## run quality-attribute test suite
	pytest tests/ -v --tb=short

gates:            ## business KPI gate
	python training/train.py

repro:            ## dvc repro = full reproducible pipeline
	dvc repro && dvc metrics show

docker:           ## build + run the scoring API
	docker build -t fraud-api .
	docker run -p 8000:8000 fraud-api
