import requests
 
bearer_token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiaWpheTU2MjciLCJpZCI6MiwiZXhwIjoxNzE3NTMwMDYyfQ.OwFF1h10bKV-REkOCtsGYYxf27ngIPNaQoFvkVWU5Ss'
 
header = {
    "Authorization": f"Bearer {bearer_token}"
}
 
# Create form data with only the prompt field
form_data = {
    "prompt": (None, "Who are you?")
}
 
# Send the request with form data
response = requests.post(
    "https://spiralogics.ai/samchat/ask",
    files=form_data,  # Use files to send multipart/form-data
    headers=header
)
 
# Check the response
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(response.content)