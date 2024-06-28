from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt, JWTError
from db import database
from starlette import status
from typing import Annotated
from datetime import timedelta, datetime
from limiter import limiter

router = APIRouter(
    prefix='/auth',
    tags = ['auth']
)

SECRET_KEY = '197b2c37c391b3d93fe80344fe73b806947a65e37297d05a1a23cffa12702fe3'
ALGORITH = 'HS256'

oauth2_bearer = OAuth2PasswordBearer(tokenUrl = 'auth/token')

class Token(BaseModel):
    access_token: str
    token_type: str

@router.post('/token', response_model= Token)
@limiter.limit("50/second")
async def login_for_access_token(request: Request, form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user, user_id = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail='could not validate user')
    token = create_access_token(user, user_id, timedelta(minutes = 720))

    return {'access_token': token, 'token_type': 'bearer'}
    
def create_access_token(username: str, user_id:int, expires_delta: timedelta):
    encode = {'sub':username, 'id':user_id}
    expires = datetime.now() + expires_delta
    encode.update({'exp': expires}) 
    return jwt.encode(encode, SECRET_KEY, algorithm=ALGORITH)


#decode the jwt
async def get_current_user(token: Annotated[str, Depends(oauth2_bearer)]):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=ALGORITH)
        username: str = payload.get('sub')
        user_id: int = payload.get('id')
        if username is None or user_id is None:
            raise HTTPException(status_code= status.HTTP_401_UNAUTHORIZED, detail= ' could not validate user')
        return {'username': username, 'id': user_id}
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='could not validate user')


def authenticate_user(username: str, password: str):
    for d in database:
        if username == d['username'] and password == d['password']:
            return username, d['id']
    return False
    
