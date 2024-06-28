import requests
import base64

encoded_string = None

with open("E:\\ProjectX\\tmp\\medical.pdf", "rb") as pdf_file:
    encoded_string = base64.b64encode(pdf_file.read()).decode("utf-8")

header={"x-api-key" : 'e54d4431-5dab-474e-b71a-0db1fcb9e659'}

response = requests.post(
    "http://localhost:8660/ask_from_new_document",
    json={"pdf_name": "medical.pdf","pdf_base64": encoded_string, "prompt": "who are you?", "cntxt_key":"P2R"},
    headers=header
)

if response.status_code == 200:
    data = response.json()

    print(data)
else:
    print(response.content)
