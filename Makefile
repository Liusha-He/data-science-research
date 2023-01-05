SHELL := /bin/bash

init:
	pip install poetry
	poetry install

compile:
	poetry update

start:
	jupyter notebook

.PHONY: init compile start
