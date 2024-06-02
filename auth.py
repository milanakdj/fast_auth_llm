from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt, JWTError
from db import database
from starlette import status
from typing import Annotated
from datetime import timedelta, datetime


router = APIRouter(
    prefix='/auth',
    tags = ['auth']
)

SECRET_KEY = '197b2c37c391b3d93fe80344fe73b806947a65e37297d05a1a23cffa12702fe3'
ALGORITH = 'HS256'

bcrypt_context = CryptContext(schemes= ['bcrypt'], deprecated = 'auto')
oauth2_bearere = OAuth2PasswordBearer(tokenUrl = 'auth/token')

class Token(BaseModel):
    access_token: str
    token_type: str

def get_db():
    user_db = database
    yield user_db

# db_dependency = Annotated[, Depends(get_db)]
    

@router.post('/token', response_model= Token)
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user, user_id = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail='could not validate user')
    token = create_access_token(user, user_id, timedelta(minutes = 60))

    return {'access_token': token, 'token_type': 'bearer'}
    
def create_access_token(username: str, user_id:int, expires_delta: timedelta):
    encode = {'sub':username, 'id':user_id}
    expires = datetime.now() + expires_delta
    encode.update({'exp': expires}) 
    return jwt.encode(encode, SECRET_KEY, algorithm=ALGORITH)

def authenticate_user(username: str, password: str):
    if username == database['username'] and password == database['password']:
        return username, database['id']
    else:
        return False
    
