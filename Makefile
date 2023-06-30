.PHONY: start
start:
	uvicorn main:app --reload --port 8081

.PHONY: format
format:
	black .
	isort .