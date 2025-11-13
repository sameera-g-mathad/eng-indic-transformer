# This file contains common commands to run.

install:
	pip3 install .

install_dev:
	pip3 install .[dev]

pipreqs:
	pipreqs . --scan-notebooks --savepath  requirements/requirements.txt 

pipreqs_force:
	pipreqs . --force --scan-notebooks --savepath  requirements/requirements.txt 