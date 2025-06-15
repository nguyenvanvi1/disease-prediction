import os
import io
import json
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, UnidentifiedImageError
from typing import List, Optional
from jose import JWTError, jwt
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table, Float, DateTime
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
load_dotenv()
import os
SECRET_KEY = os.getenv("SECRET_KEY", "default-key-if-not-set")
DATABASE_URL = os.getenv("DATABASE_URL", "default-key-if-not-set")


app = FastAPI(
    title="Plant Disease Classification API",
    description="API for classifying plant diseases with user authentication",
    version="1.0.0"
)

# --- CẤU HÌNH ---
MODEL_PATH = "mobilenet_model.h5"
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]
IMAGE_SIZE = (224, 224)


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
metadata = MetaData()

users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("name", String, unique=True, index=True),
    Column("hashed_password", String),
)

history = Table(
    "history",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("user_id", Integer),
    Column("name", String, nullable=True),
    Column("confidence", Float),
    Column("description", String, nullable=True),
    Column("symptoms", String, nullable=True),
    Column("impact", String, nullable=True),
    Column("solutions", String),
    Column("image_path", String, nullable=True),
    Column("timestamp", DateTime),
)

metadata.create_all(bind=engine)

  # Thay bằng key mạnh (openssl rand -hex 32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Tạo thư mục uploads nếu chưa tồn tại
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load disease_info.json
DISEASE_INFO = {}
try:
    with open("disease_info.json", "r", encoding="utf-8") as f:
        DISEASE_INFO = json.load(f)
    print("✅ Loaded disease_info.json successfully")
except FileNotFoundError:
    print("⚠️ Warning: disease_info.json not found, using empty DISEASE_INFO.")
except json.JSONDecodeError as e:
    print(f"⚠️ Warning: Invalid JSON in disease_info.json: {e}, using empty DISEASE_INFO.")

# Load model
model = None
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {str(e)}")
    raise RuntimeError("Cannot start server without model")

async def preprocess_image(file: UploadFile) -> np.ndarray:
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File is not an image!")
        contents = await file.read()
        try:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image file!")
        img = img.resize(IMAGE_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, contents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db=Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        name: str = payload.get("sub")
        if name is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    query = users.select().where(users.c.name == name)
    user = db.execute(query).fetchone()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

class UserRegister(BaseModel):
    name: str
    password: str

class UserLogin(BaseModel):
    name: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class HistoryItem(BaseModel):
    id: int
    name: str
    confidence: float
    description: Optional[str] = None
    symptoms: Optional[List[str]] = None
    impact: Optional[str] = None
    solutions: Optional[List[str]] = None
    image_path: Optional[str] = None
    timestamp: datetime

@app.post("/register")
async def register_user(user: UserRegister, db=Depends(get_db)):
    query = users.select().where(users.c.name == user.name)
    if db.execute(query).fetchone():
        raise HTTPException(status_code=400, detail="Tên đã tồn tại")
    hashed_password = pwd_context.hash(user.password)
    db.execute(
        users.insert().values(
            name=user.name,
            hashed_password=hashed_password,
        )
    )
    db.commit()
    return {"message": "Đăng ký thành công"}

@app.post("/login", response_model=Token)
async def login_user(data: UserLogin, db=Depends(get_db)):
    query = users.select().where(users.c.name == data.name)
    user = db.execute(query).fetchone()
    if not user:
        raise HTTPException(status_code=400, detail="Tên không tồn tại")
    if not pwd_context.verify(data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Mật khẩu không đúng")
    access_token = create_access_token(data={"sub": user.name})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict", response_model=dict)
async def predict(file: UploadFile = File(...), user=Depends(get_current_user), db=Depends(get_db)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        img_array, img_contents = await preprocess_image(file)
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # Lấy thông tin chi tiết từ DISEASE_INFO
        disease_details = DISEASE_INFO.get(predicted_class, {
            "name": predicted_class,
            "description": "Không có thông tin mô tả.",
            "symptoms": ["Không có triệu chứng được ghi nhận."],
            "solutions": ["Không có giải pháp cụ thể."],
            "impact": "Không có thông tin về tác động."
        })
        
        # Chuyển danh sách symptoms sang chuỗi JSON
        symptoms_json = json.dumps(disease_details["symptoms"])
        
        # Lưu ảnh vào thư mục uploads
        image_filename = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join(UPLOAD_DIR, image_filename)
        with open(image_path, "wb") as f:
            f.write(img_contents)
        
        # Lưu vào history
        timestamp = datetime.utcnow() + timedelta(hours=7)
        db.execute(
            history.insert().values(
                user_id=user.id,
                name=disease_details["name"],
                confidence=confidence,
                description=disease_details["description"],
                symptoms=symptoms_json,
                impact=disease_details["impact"],
                solutions=json.dumps(disease_details["solutions"]),
                image_path=image_path,
                timestamp=timestamp
            )
        )
        db.commit()
        
        return {
            "name": disease_details["name"],
            "confidence": confidence,
            "description": disease_details["description"],
            "symptoms": disease_details["symptoms"],
            "solutions": disease_details["solutions"],
            "impact": disease_details["impact"],
            "image_path": image_path,
            "message": "Success"
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/history", response_model=List[HistoryItem])
async def get_history(user=Depends(get_current_user), db=Depends(get_db)):
    query = history.select().where(history.c.user_id == user.id).order_by(history.c.timestamp.desc())
    results = db.execute(query).fetchall()
    return [
        {
            "id": row.id,
            "name": row.name,
            "confidence": row.confidence,
            "description": row.description,
            "symptoms": json.loads(row.symptoms) if row.symptoms else [],
            "impact": row.impact,
            "solutions": json.loads(row.solutions) if row.solutions else [],
            "image_path": row.image_path,
            "timestamp": row.timestamp
        }
        for row in results
    ]

@app.delete("/history/delete/{id}")
async def delete_history(id: int, user=Depends(get_current_user), db=Depends(get_db)):
    query = history.select().where(history.c.id == id)
    record = db.execute(query).fetchone()
    
    if not record:
        raise HTTPException(status_code=404, detail="Bản ghi lịch sử không tồn tại")
    
    if record.user_id != user.id:
        raise HTTPException(status_code=403, detail="Bạn không có quyền xóa bản ghi này")
    
    if record.image_path and os.path.exists(record.image_path):
        try:
            os.remove(record.image_path)
        except Exception as e:
            print(f"⚠️ Warning: Không thể xóa file ảnh {record.image_path}: {str(e)}")
    
    delete_query = history.delete().where(history.c.id == id)
    db.execute(delete_query)
    db.commit()
    
    return {"message": "Xóa lịch sử thành công"}

@app.get("/class_names", response_model=List[str])
async def get_class_names():
    return CLASS_NAMES

@app.get("/")
async def health_check():
    return {"status": "OK", "message": "Plant Disease Classification API is running"}

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)