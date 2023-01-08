SHELL := /bin/bash

init:
	pip install poetry
	poetry install

compile:
	poetry update

start:
	jupyter notebook

test:
	pytest -svv tests

.PHONY: init compile start test
