run:
	streamlit run front_end/app.py

train:
	python -m illuminating.interface.main

.PHONY: run train
