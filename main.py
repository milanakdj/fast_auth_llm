from fastapi import FastAPI
import uvicorn
import sam
import auth

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    return {'message': 'OK'}

@app.post("/query")
async def ask(prompt: str):
    #call the ai

    #send the query

    #get the response
    return (prompt)

app.include_router(sam.router)
app.include_router(auth.router)
if __name__ == '__main__':
    uvicorn.run(app, host = 'localhost', port = 8502)