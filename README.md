DDoS Detection and AI Analysis 


--- Basic Overview ---

This project is aimed at looking at AI-powered DDoS Detection tool that analyses network traffic and then classifies it as normal or attack using machine learning and running it under Python

-> Offline dataset analysis 

-> ClI-based detection tool

-> Simulated real-time traffic detection, possibly using wireshark packets. (Currently working under it,)




--- What is the Purpose of this? --- 

Distributed Denial-of-Service (DDoS) attacks are a major cybersecurity threat.
This project demonstrates how machine learning can be used to detect malicious traffic patterns in network data.



--- Features ---

Train ML model using network traffic datasets

Detect attacks via terminal (Linux CLI tool)

Analyze CSV-based traffic logs

Generate test traffic samples (attack / normal / mixed)

Output detection summary and verdict

Export results to CSV

Detection output now includes:

- traffic ratios for normal vs attack rows
- a simple terminal risk meter
- an attack-type breakdown for detected malicious traffic


--- Terminal Usage ---

This project can be run with `make` from the project root.

Install dependencies:

```bash
make install
```

Adapt a raw dataset:

```bash
make adapt
```

Train the model:

```bash
make train
```

Run detection:

```bash
make detect
```

Run the full pipeline:

```bash
make all
```

Useful overrides:

```bash
make adapt RAW_INPUT=data/raw/custom.csv ADAPTED_OUTPUT=data/raw/custom_adapted.csv
make detect DETECT_FILE=data/raw/custom_adapted.csv ROWS=50000 CHUNK_SIZE=10000
make detect OUTPUT=results.csv
```

You can also use the shell wrapper:

```bash
chmod +x run.sh
./run.sh adapt
./run.sh train
./run.sh detect
./run.sh all
```



--- Features ---

Train ML model using network traffic datasets

Detect attacks via terminal (Linux CLI tool)

Analyze CSV-based traffic logs

Generate test traffic samples (attack / normal / mixed)




--- Technologies Used ---

Python

Pandas / NumPy

Scikit-learn (Random Forest)

Git / GitHub
Linux terminal

Output detection summary and verdict

Export results to CSV
