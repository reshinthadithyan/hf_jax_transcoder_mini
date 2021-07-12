.PHONY:help
help:
	@echo "prepare-data : preprocess data"

.PHONY:prepare-data
prepare-data:
	pip install -r requirements.txt --upgrade
	python download_repo_info.py
	python download_and_process