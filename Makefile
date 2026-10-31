.PHONY: install test serve demo sweep figures clean

install:
	pip install -r requirements.txt

test:
	python tests/test_vigil.py
	python tests/test_events.py

serve:
	uvicorn server:app --reload --port 8000

demo:
	python experiments/run_experiment.py --input sample/demo_clip.avi \
		--runs A B C D --no-audio --speed 8

sweep:
	python experiments/run_experiment.py --input sample/demo_clip.avi \
		--sweep --no-audio --speed 8

figures:
	python experiments/analyze.py --log-dir logs \
		--annotations sample/annotations.csv --out-dir figures

clean:
	rm -rf logs/*.csv logs/*.json figures __pycache__ */__pycache__
