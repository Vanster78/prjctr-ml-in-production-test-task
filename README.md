<h3>Setup</h3>

```pip3 install -r requirements.txt```

<h3>Training</h3>

```python3 train.py```

<h3>Results</h3>

```
Train RMSE: 0.39
Val RMSE: 0.59
```

Model checkpoint is available here: https://drive.google.com/file/d/1rh4agioPifxDzk3ctXnRz5LUTdhWOA5j/view?usp=sharing

<h3>Running API</h3>

```
gdown 1rh4agioPifxDzk3ctXnRz5LUTdhWOA5j
mkdir training_results
unzip checkpoint-852.zip -d training_results
uvicorn main:app
```

Request example:

```python3 make_request.py```


