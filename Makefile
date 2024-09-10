.ONESHELL:
.PHONY: uninstall
.PHONY: install
.PHONY: johnny

uninstall:
	conda env remove -n quants-lab

install:
	conda env create -f environment.yml

johnny:
	nohup python3 research_notebooks/xtreet_bb/server.py > server_output.log 2>&1 &