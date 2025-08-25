from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from passlib.context import CryptContext
import cloudinary
import cloudinary.uploader
import cloudinary.api
from cloudinary.exceptions import Error as CloudinaryError
from jose import JWTError, jwt
from datetime import datetime, timedelta, UTC
from dotenv import load_dotenv
import os
from typing import Optional, List
import tempfile
import subprocess
import json
import asyncio
from pathlib import Path

load_dotenv()

# Configuration
SECRET_KEY = "your-secret-key-here-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
DATABASE_URL = os.getenv("DATABASE_URL")
UPLOAD_DIRECTORY = "uploads"

os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

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
    color_palette: List[ColorInfo]  # Changed from 'palette' to match your data
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
    width: int  # Added width
    height: int  # Added height

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

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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
        
        cmd = ["python", str(script_path), image_path]
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        "timestamp": datetime.utcnow().isoformat()
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
                "processed_at": datetime.now(UTC).isoformat(),  # Fixed deprecation warning
                "processing_time_ms": processing_time
            },
            id=result["cloudinary"]["public_id"],
            url=result["cloudinary"]["secure_url"],
            colorName=result["colors"]["dominant_color"]["name"],
            colorHex=result["colors"]["dominant_color"]["hex"],
            width=result["cloudinary"]["width"],  # Added width from cloudinary data
            height=result["cloudinary"]["height"]  # Added height from cloudinary data
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
async def create_product(product: ProductCreate, db: Session = Depends(get_db)):
    """Create a new product with images"""
    try:
        # Insert product
        db_product = Product(
            name=product.name,
            price=int(product.price * 100),  # Convert to cents
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
                "category": product.category,
                "images": [
                    {
                        "url": image.url,
                        "id": image.id,
                        "colourHex": image.colour,
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
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)