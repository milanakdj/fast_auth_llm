from fastapi import FastAPI, Response
import sam
import auth
import uvicorn
from limiter import limiter, RateLimitExceeded, _rate_limit_exceeded_handler

 
app = FastAPI()


app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(sam.router)
app.include_router(auth.router)
 
@app.get("/")
async def root():
    content = """
<pre>
API Documentation
 
Authentication Endpoint
-----------------------
 
Endpoint: /auth/token
 
**Request:**
- Method: POST
- Content Type: x-www-form-urlencoded
- Parameters:
  - `username` (string): The user's username.
  - `password` (string): The user's password.
 
**Response:**
- On Success (Status Code: 200):
  - Content Type: application/json
  - Response Body:
    {
      "access_token": "string",
      "token_type": "string"
    }
 
- On Failure (Status Code: 401 or other error codes):
  - Content Type: application/json
  - Response Body:
    {
      "detail": "Error message"
    }
 
Ask Endpoint
------------
 
Endpoint: /ask
 
**Request:**
- Method: POST
- Content Type: form-data
- Authorization: Bearer token required
- Request Body:
  {
    "prompt": "Question"
  }
 
**Response:**
- On Success (Status Code: 200):
  - Content Type: application/json
  - Example Response:
    "Answer"
 
- On Error (Status Code: 422):
  - Content Type: application/json
  - Example Response:
    {
      "detail": [
        {
          "loc": [
            "string",
            0
          ],
          "msg": "string",
          "type": "string"
        }
      ]
    }
</pre>"""
    return Response(content=content, media_type="text/html")
 
@app.get("/health")
async def health():
    return {"message": "OK"}
 
if __name__ == "__main__":
    app.run()
    #uvicorn.run(app, host="localhost", port=8590)