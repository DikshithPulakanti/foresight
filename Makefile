.PHONY: up down logs build test lint clean health

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

build:
	docker-compose build --no-cache

test:
	pytest tests/ --tb=short -v

lint:
	ruff check services/

clean:
	docker-compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

health:
	curl -s http://localhost:8000/health | python3 -m json.tool
