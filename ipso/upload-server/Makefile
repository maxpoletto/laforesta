.PHONY: test run clean

test:
	python3 -m unittest -v test_server.py

# Run against config.example.json for local hand-testing (uploads to /tmp).
run:
	python3 server.py config.example.json

clean:
	find . -name __pycache__ -type d -exec rm -rf {} +
