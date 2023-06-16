.PHONY: test
SHELL := /bin/bash

default: help

#help: @ Shows help topics
help:
	@grep -E '[a-zA-Z\.\-]+:.*?@ .*$$' $(MAKEFILE_LIST)| tr -d '#'  | awk 'BEGIN {FS = ":.*?@ "}; {printf "\033[32m%-30s\033[0m %s\n", $$1, $$2}'

#test: @ Run all tests
test:
	source .env && python -m unittest discover -v -p "test_*.py"

#lint: @ Run flake8 linter (does not format code)
lint:
	python -m flake8 --config=.flake8 --statistics --count .

#format: @ Run black code formatter
format:
	python -m black --config ./pyproject.toml .


include .env
export $(shell sed 's/=.*//' .env)

core-build:
	docker-compose build ia2-core

core-run:
	docker-compose run ia2-core

jupyter-build: core-build
	docker-compose build ia2-jupyter

jupyter-run: jupyter-build
	docker-compose up ia2-jupyter-gpu

jupyter-run-cpu: jupyter-build
	docker-compose up ia2-jupyter-cpu

core-test-cpu-all: export TEST_COMMAND=$(BASE_TEST_COMMAND)
core-test-cpu-all: core-build
	docker-compose run ia2-core-test-cpu