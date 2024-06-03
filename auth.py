from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt, JWTError
# from db import database
from starlette import status
from typing import Annotated
from datetime import timedelta, datetime
from sqlalchemy.orm import Session
from db import SessionLocal
from models import Users


router = APIRouter(
    prefix='/auth',
    tags = ['auth']
)

SECRET_KEY = '197b2c37c391b3d93fe80344fe73b806947a65e37297d05a1a23cffa12702fe3'
ALGORITH = 'HS256'

bcrypt_context = CryptContext(schemes= ['bcrypt'], deprecated = 'auto')
oauth2_bearer = OAuth2PasswordBearer(tokenUrl = 'auth/token')

class Token(BaseModel):
    access_token: str
    token_type: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]
    
class CreateUserRequest(BaseModel):
    username:str
    password:str


@router.post("/", status_code = status.HTTP_201_CREATED)
async def create_user(db:db_dependency, create_user_request: CreateUserRequest):
    create_user_model = Users(
        username = create_user_request.username,
        hashed_password = bcrypt_context.hash(create_user_request.password)
    )
    db.add(create_user_model)
    db.commit()

@router.post('/token', response_model= Token)
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db:db_dependency):
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail='could not validate user')
    token = create_access_token(user.username, user.id, timedelta(minutes = 60))

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


def authenticate_user(username: str, password: str, db):
    user = db.query(Users).filter(Users.username == username).first()
    if not user:
        return False
    if not bcrypt_context.verify(password, user.hashed_password):
        return False
    return user
    
