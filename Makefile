build_dev_requirements: 
	pip-compile requirements.in

install_dev_requirements: 
	pip install -r requirements.txt

install: 
	pip install -e .
