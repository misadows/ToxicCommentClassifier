import requests
import json

url = 'http://localhost:8501/v1/models/saved_model_basic:predict'

data = {"instances": [3.0, 2.0, 5.0]}

r = requests.post(url, data=json.dumps(data))
print(r.text)
