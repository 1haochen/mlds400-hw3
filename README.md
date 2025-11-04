# Titanic Machine Learning Pipeline

This project builds a logistic regression model to predict Titanic passenger survival using both Python and R Dockerized workflows.

---

## Before You Begin

1. Download the dataset from [Kaggle Titanic Competition Data](https://www.kaggle.com/competitions/titanic/data).
2. Place the following files under the `src/data` directory:

   ```
   src/data/train.csv
   src/data/test.csv
   ```

---

## Python Workflow

### Build and Run with Docker

```bash
# Build the Docker image
docker build -t mlds400-hw3-python -f Dockerfile_py .

# Run the container
docker run --rm mlds400-hw3-python

# Run and mount the current directory to save outputs locally
docker run --rm -v "$(pwd):/app" mlds400-hw3-python
```

### Description

* Script: `src/py-code/main.py`
* Input: `src/data/train.csv` and `src/data/test.csv`
* Output: `test_predictions-py.csv` (saved in the project root)

---

## R Workflow

### Build and Run with Docker

```bash
# Build the Docker image
docker build -t mlds400-hw3-r -f Dockerfile_R .

# Run the container
docker run --rm mlds400-hw3-r

# Run and mount the current directory to save outputs locally
docker run --rm -v "$(pwd):/app" mlds400-hw3-r
```

### Description

* Script: `src/r-code/main.r`
* Input: `src/data/train.csv` and `src/data/test.csv`
* Output: `test_predictions-r.csv` (saved in the project root)

---

## Output Files

After running either workflow, prediction results will appear in the project root:

* `test_predictions-py.csv` — Python output
* `test_predictions-r.csv` — R output

---

## Notes

* Ensure the dataset files exist before building or running the containers.
* The `--rm` flag removes containers automatically after execution.
* Mounting (`-v "$(pwd):/app"`) allows saving outputs directly to your local machine.

