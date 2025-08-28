from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form, Query, Request, Response, Cookie
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
import cloudinary
import cloudinary.uploader
import cloudinary.api
from cloudinary.exceptions import Error as CloudinaryError
from jose import JWTError, jwt
from datetime import datetime, timedelta, UTC, timezone
from dotenv import load_dotenv
import os
from typing import Optional, List, Dict, Any
import tempfile
import subprocess
import json
import asyncio
from pathlib import Path
from collections import defaultdict
import time
from functools import wraps
import ipaddress

load_dotenv()

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY")
origins = [
    "http://localhost:3000",
    "https://dfootprint-website.vercel.app",
]
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
ENVIRONMENT = os.getenv("ENVIRONMENT")
DATABASE_URL = os.getenv("DATABASE_URL")
UPLOAD_DIRECTORY = "uploads"
IS_PROD = os.getenv("ENV") == "production"

def get_start_command() -> str:
    if ENVIRONMENT == "production":
        return "python"
    else : 
        return "./venv/scripts/python"

os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Rate limiting storage (in production, use Redis)
rate_limit_storage = defaultdict(list)
RATE_LIMIT_REQUESTS = 5  # 5 attempts
RATE_LIMIT_WINDOW = 900  # 15 minutes in seconds

# Cloudinary configuration
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_NAME"),
    api_key=os.getenv("CLOUDINARY_KEY"),
    api_secret=os.getenv("CLOUDINARY_SECRET")
)

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, index=True)
    slug = Column(String(255), unique=True, index=True)
    name = Column(String(255), index=True)
    description = Column(Text)
    sizes = Column(String(50), nullable=True)
    featured = Column(Boolean, default=False)
    category = Column(String(50), index=True)
    price = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class Product_image(Base):
    __tablename__ = "product_images"
    id = Column(String(255), primary_key=True, index=True)
    slug = Column(String(255), unique=True, index=True)
    product_id = Column(Integer, nullable=False)
    url = Column(String(255), nullable=False)
    colour = Column(String(50), nullable=False)
    width = Column(Integer)
    height = Column(Integer)
    color_name = Column(String(50), nullable=False)

class Admins(Base):
    __tablename__ = "admins"
    id = Column(Integer, primary_key=True, index=True)
    fullname = Column(String(255))
    email = Column(String(255), unique=True, index=True)
    role = Column(String(50), default="admin")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    password = Column(String(255))
    permissions = Column(Text)

Base.metadata.create_all(bind=engine)

# Pydantic Models
class ColorInfo(BaseModel):
    rank: int
    rgb: List[int]
    hex: str
    name: str

class DominantColor(BaseModel):
    rank: int
    rgb: List[int]
    hex: str
    name: str

class ColorData(BaseModel):
    success: bool
    dominant_color: DominantColor
    color_palette: List[ColorInfo]
    image_processed: str

class CloudinaryResult(BaseModel):
    public_id: str
    secure_url: str
    url: str
    width: int
    height: int
    format: str
    bytes: int
    created_at: str

class ImageProcessingResult(BaseModel):
    success: bool
    colors: ColorData
    cloudinary: CloudinaryResult
    filename: str

class Image_Data(BaseModel):
    id: str
    slug: str
    product_id: int
    url: str
    colour: str
    width: int
    height: int
    color_name: str

class CreateProduct(BaseModel):
    slug: str
    name: str
    description: str
    sizes: str
    featured: bool
    category: str
    price: int
    image_data: Image_Data

class ProductResponse(BaseModel):
    id: int
    slug: str
    name: str
    description: str
    sizes: list[str]
    featured: bool
    category: str
    price: int
    image_data: Image_Data

class ImageUploadResponse(BaseModel):
    success: bool
    colors: ColorData
    image: dict
    metadata: dict
    id: str
    url: str
    colorName: str
    colorHex: str
    width: int
    height: int

class BatchResult(BaseModel):
    filename: str
    success: bool
    colors: Optional[ColorData] = None
    cloudinary: Optional[CloudinaryResult] = None
    error: Optional[str] = None

class BatchUploadResponse(BaseModel):
    success: bool
    processed: int
    successful: int
    failed: int
    processing_time_ms: int
    results: List[BatchResult]

class ProductCreate(BaseModel):
    name: str
    price: float
    description: str
    sizes: list[str]
    featured: bool
    category: str
    images: Optional[List[dict]] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class FileUploadResponse(BaseModel):
    id: int
    filename: str
    original_filename: str
    file_size: int
    uploaded_at: datetime

# Authentication Models
class AdminLogin(BaseModel):
    email: EmailStr
    password: str

class AdminCreate(BaseModel):
    fullname: str
    email: EmailStr
    password: str
    role: str = "admin"
    permissions: List[str] = []

class AdminResponse(BaseModel):
    id: int
    fullname: str
    email: str
    role: str
    permissions: List[str]
    created_at: datetime
    updated_at: datetime

class TokenData(BaseModel):
    email: Optional[str] = None

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def decode_access_token(token: str) -> Optional[dict]:
    try:
        # decode and validate exp claim automatically
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_client_ip(request: Request) -> str:
    """Get client IP address, considering proxy headers"""
    # Check for Render.com forwarded IP
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    # Check for real IP
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to client host
    return request.client.host if request.client else "unknown"

def is_rate_limited(ip: str) -> bool:
    """Check if IP is rate limited"""
    now = time.time()
    
    # Clean old entries
    rate_limit_storage[ip] = [
        timestamp for timestamp in rate_limit_storage[ip]
        if now - timestamp < RATE_LIMIT_WINDOW
    ]
    
    # Check if rate limit exceeded
    if len(rate_limit_storage[ip]) >= RATE_LIMIT_REQUESTS:
        return True
    
    # Add current attempt
    rate_limit_storage[ip].append(now)
    return False

def rate_limit_decorator():
    """Decorator for rate limiting endpoints"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find request object in args
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if request:
                client_ip = get_client_ip(request)
                if is_rate_limited(client_ip):
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail=f"Too many login attempts. Try again in {RATE_LIMIT_WINDOW//60} minutes."
                    )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(
    access_token: str = Cookie(None),
    db: Session = Depends(get_db)
):
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = decode_access_token(access_token)  # your JWT decode
    email = payload.get("sub")
    admin = db.query(Admins).filter(Admins.email == email).first()
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return admin

# Utility functions
async def upload_to_cloudinary(image_path: str, options: dict = None) -> dict:
    """Upload image to Cloudinary without transformations"""
    try:
        upload_options = {
            "folder": "fashion-colors",
            "resource_type": "image"
        }
        
        if options:
            upload_options.update(options)
        
        unique_id = f"fashion-{int(datetime.now().timestamp())}-{os.urandom(4).hex()}"
        upload_options["public_id"] = unique_id
        
        result = cloudinary.uploader.upload(image_path, **upload_options)
        
        cloudinary_response = {
            "public_id": result["public_id"],
            "secure_url": result["secure_url"],
            "url": result["url"],
            "width": result["width"],
            "height": result["height"],
            "format": result["format"],
            "bytes": result["bytes"],
            "created_at": result["created_at"]
        }
        
        return cloudinary_response
        
    except Exception as e:
        raise e
 
async def extract_colors(image_path: str, remove_background: bool = True) -> dict:
    """Extract colors from image using Python script"""
    try:
        script_path = Path(__file__).parent / "color_extractor.py"
        
        if not script_path.exists():
            raise FileNotFoundError(f"color_extractor.py not found at {script_path}")

        cmd = [get_start_command(), str(script_path), image_path]
        if not remove_background:
            cmd.append("--keep-background")
        
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=50
        )
        
        if process.returncode != 0:
            raise RuntimeError(f"Colour extraction script failed: {process.stderr}")
        
        if not process.stdout.strip():
            raise ValueError("Empty output from Colour extraction script")
            
        try:
            result = json.loads(process.stdout)
            return result
        except json.JSONDecodeError as json_error:
            raise json_error
        
    except subprocess.TimeoutExpired:
        raise TimeoutError("Colour extraction timed out")
    except Exception as e:
        raise e

async def process_image(image_path: str, filename: str, remove_background: bool = True) -> dict:
    """Process single image: extract colors and upload to Cloudinary"""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        color_task = extract_colors(image_path, remove_background)
        cloudinary_task = upload_to_cloudinary(image_path)
        
        try:
            color_data, cloudinary_result = await asyncio.gather(color_task, cloudinary_task)
        except Exception as gather_error:
            raise gather_error
        
        result = {
            "success": True,
            "colors": color_data,
            "cloudinary": cloudinary_result,
            "filename": filename
        }
        
        return result
        
    except Exception as process_error:
        raise process_error

# FastAPI app
app = FastAPI(title="FastAPI with MySQL, Authentication, and Image Processing", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,         # 👈 explicitly allow Next.js
    allow_credentials=True,        # 👈 allow cookies/auth headers
    allow_methods=["*"],           # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],           # Allow all headers (Authorization, Content-Type, etc.)
)


# Routes
@app.get("/")
async def root():
    return {"message": "FastAPI with MySQL, Authentication, and Image Processing"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Fashion Colour Extractor API",
        "timestamp": datetime.now(UTC).isoformat()
    }

# Authentication Routes
@app.post("/admin/login", response_model=Token)
@rate_limit_decorator()
async def login_admin(
    request: Request,
    admin_data: AdminLogin,
    db: Session = Depends(get_db)
):
    """Admin login with IP rate limiting"""
    
    # Get admin by email
    admin = db.query(Admins).filter(Admins.email == admin_data.email).first()
    
    if not admin or not verify_password(admin_data.password, admin.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": admin.email}, expires_delta=access_token_expires
    )

    # Set cookie
    response = JSONResponse( content = {"message": "Login successful"})
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,  # JS can't read it
        secure=IS_PROD,  # only HTTPS in prod
        samesite="strict" if IS_PROD else "lax",  # dev more lenient
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        expires=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )

    return response

@app.post("/admin/logout")
async def logout_admin():
    response = JSONResponse( content = {"message": "Logged out successfully"})
    response.delete_cookie(
        key="access_token",
        httponly=True,
        samesite="strict" if IS_PROD else "lax",  # or "strict" in prod
        secure=IS_PROD     # True in prod with HTTPS
    )
    return response

@app.get("/admin/me", response_model=AdminResponse)
async def get_current_admin(current_admin: Admins = Depends(get_current_user)):
    """Get current admin user data from token"""
    
    # Parse permissions JSON
    try:
        permissions = json.loads(current_admin.permissions) if current_admin.permissions else {}
    except json.JSONDecodeError:
        permissions = {}
    
    return AdminResponse(
        id=current_admin.id,
        fullname=current_admin.fullname,
        email=current_admin.email,
        role=current_admin.role,
        permissions=permissions,
        created_at=current_admin.created_at,
        updated_at=current_admin.updated_at
    )

@app.post("/admin/create", response_model=dict)
async def create_admin(
    admin_data: AdminCreate,
    db: Session = Depends(get_db),
    current_admin: Admins = Depends(get_current_user)
):
    """Create a new admin (requires authentication)"""
    
    # Check if current admin has permission to create admins
    try:
        current_permissions = json.loads(current_admin.permissions) if current_admin.permissions else {}
    except json.JSONDecodeError:
        current_permissions = {}
    
    # Check if user has admin creation permission or is super admin
    if not ("edit_product" in current_permissions or current_admin.role == "super admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to create admin"
        )
    
    # Check if admin with email already exists
    existing_admin = db.query(Admins).filter(Admins.email == admin_data.email).first()
    if existing_admin:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin with this email already exists"
        )
    
    # Hash password
    hashed_password = get_password_hash(admin_data.password)
    
    # Create new admin
    db_admin = Admins(
        fullname=admin_data.fullname,
        email=admin_data.email,
        password=hashed_password,
        role=admin_data.role,
        permissions=json.dumps(admin_data.permissions),
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC)
    )
    
    try:
        db.add(db_admin)
        db.commit()
        db.refresh(db_admin)
        
        return {
            "success": True,
            "message": "Admin created successfully",
            "admin_id": db_admin.id,
            "email": db_admin.email
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create admin"
        )

# Protected endpoint example
@app.get("/admin/stats")
async def get_admin_stats(
    current_admin: Admins = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get admin dashboard stats (protected route)"""
    
    # Check permissions
    try:
        permissions = json.loads(current_admin.permissions) if current_admin.permissions else {}
    except json.JSONDecodeError:
        permissions = {}
    
    if not ("edit_product" in permissions or current_admin.role == "super admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to view stats"
        )
    
    # Get stats
    total_products = db.query(Product).count()
    total_admins = db.query(Admins).count()
    
    return {
        "success": True,
        "stats": {
            "total_products": total_products,
            "total_admins": total_admins,
            "current_admin": current_admin.fullname,
            "current_role": current_admin.role
        }
    }

@app.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    removeBackground: bool = Form(True),
    db: Session = Depends(get_db)
):
    """Main Colour extraction endpoint with Cloudinary upload"""
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    # Check file size (10MB limit)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
    
    start_time = datetime.now()
    temp_file_path = None
    
    try:
        # Save uploaded file to temporary location
        suffix = Path(file.filename).suffix if file.filename else '.jpg'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file_path = temp_file.name
            content = await file.read()
            temp_file.write(content)
        
        # Process image (Colour extraction + Cloudinary upload)
        result = await process_image(temp_file_path, file.filename, removeBackground)
        print(result)
        
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Prepare response with width and height included
        response = ImageUploadResponse(
            success=True,
            colors=result["colors"],
            image={
                "cloudinary": result["cloudinary"],
                "original_filename": file.filename,
                "file_size": file.size,
                "processing_time_ms": processing_time,
                "remove_background": removeBackground
            },
            metadata={
                "processed_at": datetime.now(UTC).isoformat(),
                "processing_time_ms": processing_time
            },
            id=result["cloudinary"]["public_id"],
            url=result["cloudinary"]["secure_url"],
            colorName=result["colors"]["dominant_color"]["name"],
            colorHex=result["colors"]["dominant_color"]["hex"],
            width=result["cloudinary"]["width"],
            height=result["cloudinary"]["height"]
        )
        
        return response
        
    except Exception as error:
        # Convert specific errors to appropriate HTTP status codes
        if isinstance(error, FileNotFoundError):
            raise HTTPException(status_code=404, detail=f"Required file not found: {str(error)}")
        elif isinstance(error, TimeoutError):
            raise HTTPException(status_code=408, detail=f"Operation timed out: {str(error)}")
        elif isinstance(error, json.JSONDecodeError):
            print("2")
            raise HTTPException(status_code=500, detail=f"Invalid response from Colour extraction: {str(error)}")
        else:
            print(error)
            raise HTTPException(status_code=500, detail=f"Image processing failed: {str(error)}")
        
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                pass

@app.post("/upload-batch", response_model=BatchUploadResponse)
async def upload_batch(
    files: List[UploadFile] = File(...),
    removeBackground: bool = Form(True),
    db: Session = Depends(get_db)
):
    """Batch processing endpoint with Cloudinary uploads"""
    
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No image files provided")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    start_time = datetime.now()
    temp_files = []
    
    try:
        # Process all files
        processing_tasks = []
        
        for file in files:
            # Validate file
            if not file.content_type.startswith('image/'):
                processing_tasks.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Only image files are allowed"
                })
                continue
                
            if file.size > 10 * 1024 * 1024:
                processing_tasks.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "File too large. Maximum size is 10MB"
                })
                continue
            
            # Save to temp file
            suffix = Path(file.filename).suffix if file.filename else '.jpg'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_files.append(temp_file.name)
                
                processing_tasks.append({
                    "temp_path": temp_file.name,
                    "filename": file.filename,
                    "original_size": file.size
                })
        
        # Process all valid images concurrently
        results = []
        for task in processing_tasks:
            if "error" in task:
                results.append(BatchResult(
                    filename=task["filename"],
                    success=False,
                    error=task["error"]
                ))
            else:
                try:
                    result = await process_image(task["temp_path"], task["filename"], removeBackground)
                    results.append(BatchResult(
                        filename=task["filename"],
                        success=True,
                        colors=result["colors"],
                        cloudinary=result["cloudinary"]
                    ))
                except Exception as e:
                    results.append(BatchResult(
                        filename=task["filename"],
                        success=False,
                        error=str(e)
                    ))
        
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        success_count = sum(1 for r in results if r.success)
        failure_count = len(results) - success_count
        
        return BatchUploadResponse(
            success=True,
            processed=len(results),
            successful=success_count,
            failed=failure_count,
            processing_time_ms=processing_time,
            results=results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up all temp files
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                pass

@app.post("/products")
async def create_product(
    product: ProductCreate, 
    db: Session = Depends(get_db),
    current_admin: Admins = Depends(get_current_user)
):
    """Create a new product with images (protected)"""
    
    # Check permissions
    try:
        permissions = json.loads(current_admin.permissions) if current_admin.permissions else {}
    except json.JSONDecodeError:
        permissions = {}
    
    if not ("edit_product" in permissions or current_admin.role == "super admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to create product"
        )
    
    try:
        # Insert product
        db_product = Product(
            name=product.name,
            price=int(product.price),  # Convert to cents
            description=product.description,
            featured=product.featured,
            category=product.category,
            sizes=json.dumps(product.sizes),
            slug=f"{product.name.lower().replace(' ', '-')}-{int(datetime.now().timestamp())}"
        )
        db.add(db_product)
        db.commit()
        db.refresh(db_product)
        
        # Insert images if provided
        if product.images:
            for image in product.images:
                db_image = Product_image(
                    id=image["id"],
                    product_id=db_product.id,
                    url=image["url"],
                    colour=image["colourHex"],
                    color_name=image["colourName"],
                    width=image.get("width"),
                    height=image.get("height"),
                    slug=f"{db_product.name}-{image['colourName']}"
                )
                db.add(db_image)
            
            db.commit()
        
        return {
            "success": True,
            "message": "Product created successfully",
            "productId": db_product.id
        }
        
    except Exception as e:
        db.rollback()
        print("Error creating product:", e)
        raise HTTPException(status_code=500, detail="Failed to create product")

@app.get("/products")
async def get_products(
    page: int = Query(1, ge=1, description="Page number starting from 1"),
    limit: int = Query(12, ge=1, le=100, description="Number of products per page"),
    db: Session = Depends(get_db)
):
    """Get products with pagination"""
    try:
        # Calculate offset
        offset = (page - 1) * limit
        
        # Get total count for pagination info
        total_products = db.query(Product).count()
        
        # Get paginated products
        products = db.query(Product)\
            .offset(offset)\
            .limit(limit)\
            .all()
        
        result = []
        for product in products:
            images = db.query(Product_image)\
                .filter(Product_image.product_id == product.id)\
                .all()
            
            product_data = {
                "id": product.id,
                "name": product.name,
                "price": product.price,
                "description": product.description,
                "sizes": json.loads(product.sizes) if product.sizes else [],
                "featured": product.featured,
                "slug": product.slug,
                "category": product.category,
                "created_at": product.created_at,
                "new": datetime.now(timezone.utc) - (product.created_at.replace(tzinfo=timezone.utc) if product.created_at.tzinfo is None else product.created_at) < timedelta(days=7),
                "images": [
                    {
                        "url": image.url,
                        "id": image.id,
                        "colourHex": image.colour,
                        "slug": image.slug,
                        "colourName": image.color_name,
                        "width": image.width,
                        "height": image.height
                    }
                    for image in images
                ]
            }
            result.append(product_data)
        
        # Calculate pagination metadata
        total_pages = (total_products + limit - 1) // limit  # Ceiling division
        has_next = page < total_pages
        has_prev = page > 1
        
        return {
            "success": True,
            "products": result,
            "pagination": {
                "current_page": page,
                "per_page": limit,
                "total_products": total_products,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_prev": has_prev
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch products: {str(e)}")

@app.get("/products/{slug}")
async def get_product(slug: str, db: Session = Depends(get_db)):
    """Get a single product by slug"""
    try:
        product = db.query(Product).filter(Product.slug == slug).first()
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        images = db.query(Product_image).filter(Product_image.product_id == product.id).all()

        relatedProduct = db.query(Product).filter(Product.category == product.category, Product.id != product.id).all()

        result = []
        for related in relatedProduct:
            related_images = db.query(Product_image).filter(Product_image.product_id == related.id).all()
            result.append({
                "id": related.id,
                "name": related.name,
                "price": related.price,  # Convert back from cents
                "description": related.description,
                "sizes": json.loads(related.sizes) if related.sizes else [],
                "featured": related.featured,
                "new": datetime.now(timezone.utc) - (related.created_at.replace(tzinfo=timezone.utc) if related.created_at.tzinfo is None else related.created_at) < timedelta(days=7),
                "category": related.category,
                "slug": related.slug,
                "created_at": related.created_at,
                "images": [
                    {
                        "url": image.url,
                        "id": image.id,
                        "colourHex": image.colour,
                        "slug": image.slug,
                        "colourName": image.color_name,
                        "width": image.width,
                        "height": image.height
                    }
                    for image in related_images
                ]
            })

        return {
            "id": product.id,
            "name": product.name,
            "price": product.price,  # Convert back from cents
            "description": product.description,
            "sizes": json.loads(product.sizes) if product.sizes else [],
            "featured": product.featured,
            "new": datetime.now(timezone.utc) - (product.created_at.replace(tzinfo=timezone.utc) if product.created_at.tzinfo is None else product.created_at) < timedelta(days=7),
            "category": product.category,
            "slug": product.slug,
            "relatedProducts": result,
            "created_at": product.created_at,
            "images": [
                {
                    "id": image.id,
                    "url": image.url,
                    "colourHex": image.colour,
                    "colourName": image.color_name,
                    "slug": image.slug,
                    "width": image.width,
                    "height": image.height
                }
                for image in images
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch product: {str(e)}")

@app.delete("/products/{product_id}")
async def delete_product(
    product_id: int, 
    db: Session = Depends(get_db),
    current_admin: Admins = Depends(get_current_user)
):
    """Delete a product by ID (protected)"""
    
    # Check permissions
    try:
        permissions = json.loads(current_admin.permissions) if current_admin.permissions else {}
    except json.JSONDecodeError:
        permissions = {}
    
    if not ("edit_product" in permissions or current_admin.role == "super admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to delete product"
        )
    
    try:
        db_product = db.query(Product).filter(Product.id == product_id).first()
        if not db_product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Delete associated images first
        db.query(Product_image).filter(Product_image.product_id == product_id).delete()
        
        # Delete product
        db.delete(db_product)
        db.commit()
        
        return {"success": True, "message": "Product deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete product: {str(e)}")

@app.put("/products/{product_id}")
async def update_product(
    product_id: int, 
    product: ProductCreate, 
    db: Session = Depends(get_db),
    current_admin: Admins = Depends(get_current_user)
):
    """Update an existing product (protected)"""
    
    # Check permissions
    try:
        permissions = json.loads(current_admin.permissions) if current_admin.permissions else {}
    except json.JSONDecodeError:
        permissions = {}
    
    if not ("edit_product" in permissions or current_admin.role == "super admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to edit product"
        )
    
    try:
        # Get existing product
        db_product = db.query(Product).filter(Product.id == product_id).first()
        if not db_product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Update product fields
        db_product.name = product.name
        db_product.price = int(product.price)  # Convert to cents
        db_product.description = product.description
        db_product.featured = product.featured
        db_product.category = product.category
        db_product.sizes = json.dumps(product.sizes)
        db_product.updated_at = datetime.now(UTC)
        
        # Get current images from database
        existing_images = db.query(Product_image).filter(Product_image.product_id == product_id).all()
        existing_image_ids = {img.id for img in existing_images}
        
        # Get new image IDs from request
        new_image_ids = {img["id"] for img in product.images} if product.images else set()
        
        # Delete removed images
        images_to_delete = existing_image_ids - new_image_ids
        if images_to_delete:
            db.query(Product_image).filter(
                Product_image.product_id == product_id,
                Product_image.id.in_(images_to_delete)
            ).delete(synchronize_session=False)
        
        # Update or add new images
        if product.images:
            for image in product.images:
                existing_image = db.query(Product_image).filter(
                    Product_image.id == image["id"]
                ).first()
                
                if existing_image:
                    # Update existing image
                    existing_image.url = image["url"]
                    existing_image.colour = image["colourHex"]
                    existing_image.color_name = image["colourName"]
                    existing_image.width = image.get("width")
                    existing_image.height = image.get("height")
                else:
                    # Add new image
                    db_image = Product_image(
                        id=image["id"],
                        product_id=product_id,
                        url=image["url"],
                        colour=image["colourHex"],
                        color_name=image["colourName"],
                        width=image.get("width"),
                        height=image.get("height"),
                        slug=f"{db_product.name}-{image['colourName']}"
                    )
                    db.add(db_image)
        
        db.commit()
        
        return {"success": True, "message": "Product updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print("Error updating product:", e)
        raise HTTPException(status_code=500, detail="Failed to update product")

# Additional admin management endpoints
@app.get("/admin/list")
async def list_admins(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50),
    current_admin: Admins = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all admins (super admin only)"""

    if current_admin.role != "super admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only super admin can list all admins"
        )
    
    offset = (page - 1) * limit
    total_admins = db.query(Admins).count()
    admins = db.query(Admins).offset(offset).limit(limit).all()
    
    admin_list = []
    for admin in admins:
        try:
            permissions = json.loads(admin.permissions) if admin.permissions else {}
        except json.JSONDecodeError:
            permissions = {}
        
        admin_list.append({
            "id": admin.id,
            "fullname": admin.fullname,
            "email": admin.email,
            "role": admin.role,
            "permissions": permissions,
            "created_at": admin.created_at,
            "updated_at": admin.updated_at
        })
    
    total_pages = (total_admins + limit - 1) // limit
    
    return {
        "success": True,
        "admins": admin_list,
        "pagination": {
            "current_page": page,
            "per_page": limit,
            "total_admins": total_admins,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    }

@app.put("/admin/{admin_id}/permissions")
async def update_admin_permissions(
    admin_id: int,
    permissions: Dict[str, Any],
    current_admin: Admins = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update admin permissions (super admin only)"""

    if current_admin.role != "super admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only super admin can update permissions"
        )
    
    admin = db.query(Admins).filter(Admins.id == admin_id).first()
    if not admin:
        raise HTTPException(status_code=404, detail="Admin not found")
    
    if admin.id == current_admin.id and not permissions.get("create_admin", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot remove your own admin creation permission"
        )
    
    try:
        admin.permissions = json.dumps(permissions)
        admin.updated_at = datetime.now(UTC)
        db.commit()
        
        return {
            "success": True,
            "message": "Permissions updated successfully"
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update permissions")

@app.delete("/admin/{admin_id}")
async def delete_admin(
    admin_id: int,
    current_admin: Admins = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete admin (super admin only, cannot delete self)"""
    
    if current_admin.role != "super admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only super admin can delete admins"
        )
    
    if admin_id == current_admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    admin = db.query(Admins).filter(Admins.id == admin_id).first()
    if not admin:
        raise HTTPException(status_code=404, detail="Admin not found")
    
    try:
        db.delete(admin)
        db.commit()
        
        return {
            "success": True,
            "message": "Admin deleted successfully"
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete admin")

@app.post("/admin/change-password")
async def change_password(
    old_password: str = Form(...),
    new_password: str = Form(...),
    current_admin: Admins = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Change admin password"""
    
    # Verify old password
    if not verify_password(old_password, current_admin.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
    
    # Validate new password (you can add more validation here)
    if len(new_password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be at least 6 characters long"
        )
    
    try:
        # Update password
        current_admin.password = get_password_hash(new_password)
        current_admin.updated_at = datetime.now(UTC)
        db.commit()
        
        return {
            "success": True,
            "message": "Password changed successfully"
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to change password")

# Rate limit info endpoint (for debugging)
@app.get("/admin/rate-limit-info")
async def get_rate_limit_info(
    request: Request,
    current_admin: Admins = Depends(get_current_user)
):
    """Get rate limit information for current IP (admin only)"""
    
    if current_admin.role != "super admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only super admin can view rate limit info"
        )
    
    client_ip = get_client_ip(request)
    now = time.time()
    
    # Clean old entries for this IP
    rate_limit_storage[client_ip] = [
        timestamp for timestamp in rate_limit_storage[client_ip]
        if now - timestamp < RATE_LIMIT_WINDOW
    ]
    
    attempts_count = len(rate_limit_storage[client_ip])
    remaining_attempts = max(0, RATE_LIMIT_REQUESTS - attempts_count)
    
    # Calculate reset time
    if rate_limit_storage[client_ip]:
        oldest_attempt = min(rate_limit_storage[client_ip])
        reset_time = oldest_attempt + RATE_LIMIT_WINDOW
        time_until_reset = max(0, int(reset_time - now))
    else:
        time_until_reset = 0
    
    return {
        "success": True,
        "rate_limit_info": {
            "client_ip": client_ip,
            "attempts_in_window": attempts_count,
            "remaining_attempts": remaining_attempts,
            "window_seconds": RATE_LIMIT_WINDOW,
            "max_attempts": RATE_LIMIT_REQUESTS,
            "time_until_reset_seconds": time_until_reset,
            "is_rate_limited": remaining_attempts == 0
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
