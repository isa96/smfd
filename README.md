# ML

## Objective

The objective is to develop a simple model that can detect whether someone is wearing a mask or not. 

## Install

```
pip install -r requirements.txt
```

## Usage

1. Run **inference.py** This code aims to test the model. 
 

## API 

1. Run in command prompt

```
cd api
```

```
uvicorn endpoints:app --workers 4
```

2. FastAPI Swagger

```
http://127.0.0.1:8000/api/v1/ml/docs
```

**Run in Docker container**

1. Run in command prompt

```
cd api
```

2. Build Docker image

```
docker build --tag izzansilmi/sfmd-ml:1.0.5 .
```

3. Run Docker container

```
docker run -it --name sfmd-ml -p 7860:7860 izzansilmi/sfmd-ml:1.0.5
```
4. FastAPI Swagger

```
http://127.0.0.1:7860/api/v1/ml/docs
```


## Deploy ke Hugging Face Spaces / Remote Server

1. Buat space baru - https://huggingface.co/spaces.

2. Commit and push code to the space, follow readme instructions. Docker container will be deployed automatically. Example:

```
https://huggingface.co/spaces/isa96/fm-ml
```

3. ML API will be accessible by URL, you can get it from space info. Example:

```
https://isa96-fm-ml.hf.space/api/v1/ml/docs.
```



