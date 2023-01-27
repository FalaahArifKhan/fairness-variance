COMMIT_HASH := $(shell eval git rev-parse HEAD)

doc:
#	(cd benchmarks && python render.py)
	yamp source --out docs/api --verbose
	mkdocs build

livedoc: doc
	mkdocs serve --dirtyreload

develop:
	python ./setup.py develop
