

install:
	pip install --upgrade pip &&\
		pip install . &&\
			pip install -r test_requirements.txt

test:
	pytest -s -vv tests

test-all:
	tox

typing:
	tox -e typechecks

lint:
	tox -e stylechecks