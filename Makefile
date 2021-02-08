pylint: ## lint code with pylint
	pylint lqsvg -d similarities

pylint-test: ## lint test files with pylint
	pylint --rcfile=tests/pylintrc tests -d similarities

reorder-imports-staged:
	git diff --cached --name-only | xargs grep -rl --include "*.py" 'import' | xargs reorder-python-imports --separate-relative

mypy-staged:
	git diff --cached --name-only | xargs grep -rl --include "*.py" 'import' | xargs mypy --follow-imports silent

poetry-export:
	poetry export --dev -f requirements.txt -o requirements.txt

poetry-update:
	poetry update
	make poetry-export
	git add pyproject.toml poetry.lock requirements.txt
	git commit -s -m "chore(deps): make poetry update"
