SHELL := /bin/bash

init:
	pip install poetry
	poetry install

compile:
	poetry update

.PHONY: init compile
