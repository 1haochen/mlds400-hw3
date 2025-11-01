# Titanic Machine Learning Pipeline

## Before you begin

Please download the Titanic dataset from [Kaggle Titanic Competition Data](https://www.kaggle.com/competitions/titanic/data) and place `train.csv` and `test.csv` under the `src/data` directory.


## Download data

## Python

docker build -f Dockerfile_py -t mlds400-hw3-python .
docker run mlds400-hw3-python                        
docker run -v "$(pwd)/:/app" mlds400-hw3-python

### Python Workflow
- The Python workflow uses `src/py-code/main.py` to train a logistic regression model on Titanic data.
- Input data should be placed in `src/data/train.csv` and `src/data/test.csv`.
- The output file will be saved as `test_predictions-py.csv` in the project root.
- To run and mount the output, use the provided docker run command.

## R

docker build -f Dockerfile_R -t mlds400-hw3-r .
docker run --rm mlds400-hw3-r
docker run -v "$(pwd)/:/app" mlds400-hw3-r

### R Workflow
- The R workflow uses `src/r-code/main.r` to train a logistic regression model on Titanic data.
- Input data should be placed in `src/data/train.csv` and `src/data/test.csv`.
- The output file will be saved as `test_predictions-r.csv` in the project root.
- To run and mount the output, use the provided docker run command.

## Output Files
- After running either workflow, you will find the prediction results in the project root:
	- `test_predictions-py.csv` (Python)
	- `test_predictions-r.csv` (R)

## Notes
- Make sure your input data files are present before running the containers.
- The provided commands will build and run the containers as described above.


## R
docker build -f Dockerfile_py -t mlds400-hw3-r .
docker run mlds400-hw3-r                    
docker run -v "$(pwd)/:/app" mlds400-hw3-r

