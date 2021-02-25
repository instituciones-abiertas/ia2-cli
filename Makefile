.PHONY: test
SHELL := /bin/bash

default: help

#help: @ Shows help topics
help:
	@grep -E '[a-zA-Z\.\-]+:.*?@ .*$$' $(MAKEFILE_LIST)| tr -d '#'  | awk 'BEGIN {FS = ":.*?@ "}; {printf "\033[32m%-30s\033[0m %s\n", $$1, $$2}'

#test: @ Run all tests
test:
	source .env && python -m unittest discover -p "test_*.py"
