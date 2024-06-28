import requests
 
bearer_token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJTY2huYXBwczkxMDgiLCJpZCI6MSwiZXhwIjoxNzE5NTgxNDk1fQ.gxediSJUxHoVc3g2bzNlr-pirBdIjA50RF1XIFWmmqw'
 
header = {
    "Authorization": f"Bearer {bearer_token}"
}
 
# Create form data with only the prompt field
form_data = {
    "prompt": (None, "Who are you?")
}
 
# Send the request with form data
response = requests.post(
    "http://localhost:8599/ask",
    json={
          "prompt": "who are you?",
          "cntxt_key": "P2R"
        },
    headers=header
)

 
# Check the response
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(response.content)