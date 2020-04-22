.PHONY: test doc

test:
	python3 -m unittest

doc:
	pdoc3 --force --html -o docs lantern

