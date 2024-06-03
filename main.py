from fastapi import FastAPI, status, Depends, HTTPException
import models
from db import engine, SessionLocal
from typing import Annotated
from sqlalchemy.orm import Session
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

app.include_router(sam.router)
app.include_router(auth.router)


if __name__ == '__main__':
    uvicorn.run(app, host = 'localhost', port = 8502)