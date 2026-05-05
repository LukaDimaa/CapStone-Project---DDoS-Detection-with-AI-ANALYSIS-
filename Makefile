PYTHON ?= python

RAW_INPUT ?= data/raw/cicddos2019_dataset.csv
ADAPTED_OUTPUT ?= data/raw/adapted_cic2019.csv
DETECT_FILE ?= $(ADAPTED_OUTPUT)
ROWS ?=
CHUNK_SIZE ?= 5000
OUTPUT ?=
VERBOSE ?= 1

.PHONY: help install adapt train detect clean-results

help:
	@echo "Available targets:"
	@echo "  make install                          Install Python dependencies"
	@echo "  make adapt                            Adapt RAW_INPUT into ADAPTED_OUTPUT"
	@echo "  make train                            Train the model using src/train_model.py"
	@echo "  make detect                           Run detection on DETECT_FILE"
	@echo "  make clean-results                    Remove generated results_*.csv files"
	@echo ""
	@echo "Overridable variables:"
	@echo "  RAW_INPUT=data/raw/cicddos2019_dataset.csv"
	@echo "  ADAPTED_OUTPUT=data/raw/adapted_cic2019.csv"
	@echo "  DETECT_FILE=data/raw/adapted_cic2019.csv"
	@echo "  ROWS=50000"
	@echo "  CHUNK_SIZE=5000"
	@echo "  OUTPUT=results.csv"
	@echo "  VERBOSE=1"
	@echo ""
	@echo "Examples:"
	@echo "  make adapt"
	@echo "  make train"
	@echo "  make detect"
	@echo "  make detect ROWS=50000 CHUNK_SIZE=10000"
	@echo "  make adapt RAW_INPUT=data/raw/custom.csv ADAPTED_OUTPUT=data/raw/custom_adapted.csv"

install:
	$(PYTHON) -m pip install -r requirements.txt

adapt:
	$(PYTHON) src/dataset_adapter.py --input "$(RAW_INPUT)" --output "$(ADAPTED_OUTPUT)"

train:
	$(PYTHON) src/train_model.py

detect:
	$(PYTHON) src/detect.py --file "$(DETECT_FILE)" $(if $(ROWS),--rows $(ROWS),) --chunk-size $(CHUNK_SIZE) $(if $(OUTPUT),--output "$(OUTPUT)",) $(if $(filter 1 true TRUE yes YES,$(VERBOSE)),--verbose,)

clean-results:
	rm -f results_*.csv
