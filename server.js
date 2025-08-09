const express = require('express');
const multer = require('multer');
const path = require('path');
const db = require('./db').db; // Import the database connection
const fs = require('fs');
const { spawn } = require('child_process');
const cors = require('cors');
const { v2: cloudinary } = require('cloudinary'); // Fixed import syntax
const dotenv = require('dotenv');
dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

console.log('Environment Variables:', {
    MYSQL_HOST: process.env.MYSQL_HOST,
    MYSQL_USER: process.env.MYSQL_USER,
    MYSQL_DATABASE: process.env.MYSQL_DATABASE,
    MYSQL_PORT: process.env.MYSQL_PORT,
    CLOUDINARY_NAME: process.env.CLOUDINARY_NAME,
    CLOUDINARY_KEY: process.env.CLOUDINARY_KEY,
    CLOUDINARY_SECRET: process.env.CLOUDINARY_SECRET
});

cloudinary.config({
    cloud_name: process.env.CLOUDINARY_NAME,
    api_key: process.env.CLOUDINARY_KEY,
    api_secret: process.env.CLOUDINARY_SECRET
});

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = 'uploads';
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    // Generate unique filename
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, 'fashion-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    // Check if file is an image
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed!'), false);
    }
  }
});

// Function to upload to Cloudinary
function uploadToCloudinary(imagePath, options = {}) {
  return new Promise((resolve, reject) => {
    const uploadOptions = {
      folder: 'fashion-colors',
      resource_type: 'image',
      transformation: [
        { width: 800, height: 600, crop: 'limit' },
        { quality: 'auto' }
      ],
      ...options
    };

    cloudinary.uploader.upload(imagePath, uploadOptions, (error, result) => {
      if (error) {
        reject(new Error(`Cloudinary upload failed: ${error.message}`));
      } else {
        resolve({
          public_id: result.public_id,
          secure_url: result.secure_url,
          url: result.url,
          width: result.width,
          height: result.height,
          format: result.format,
          bytes: result.bytes,
          created_at: result.created_at
        });
      }
    });
  });
}

// Function to run Python color extraction
function extractColors(imagePath, removeBackground = true) {
  return new Promise((resolve, reject) => {
    const pythonScript = path.join(__dirname, 'color_extractor.py');
    const args = [pythonScript, imagePath];
    if (!removeBackground) args.push('--keep-background');

    const pythonProcess = spawn('python', args);
    let result = '';
    let error = '';

    pythonProcess.stdout.on('data', (data) => {
      result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      error += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error(`Python script exited with code ${code}`);
        reject(new Error(`Python script failed: ${error}`));
      } else {
        try {
          // Parse the JSON result from Python script
          const jsonResult = JSON.parse(result);
          resolve(jsonResult);
        } catch (parseError) {
          reject(new Error(`Failed to parse result: ${parseError.message}`));
        }
      }
    });

    // Handle timeout
    setTimeout(() => {
      pythonProcess.kill();
      reject(new Error('Color extraction timed out'));
    }, 50000); // 30 second timeout
  });
}

// Function to process single image (color extraction + cloudinary upload)
async function processImage(imagePath, filename, removeBackground = true) {
  try {
    // Run both operations simultaneously
    const [colorData, cloudinaryResult] = await Promise.all([
      extractColors(imagePath, removeBackground),
      uploadToCloudinary(imagePath, {
        public_id: `fashion-${Date.now()}-${Math.round(Math.random() * 1000)}`
      })
    ]);

    return {
      success: true,
      colors: colorData,
      cloudinary: cloudinaryResult,
      filename: filename
    };
  } catch (error) {
    throw new Error(`Processing failed: ${error.message}`);
  }
}

// Routes

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    service: 'Fashion Color Extractor API',
    timestamp: new Date().toISOString()
  });
});

app.post('/products', async (req, res) => {
  const { name, price, description, category, estimatedDelivery, images } = req.body;
  try {
    const [result] = await db.query(
        'INSERT INTO products (name, price, description, category, estimatedDelivery) VALUES (?, ?, ?, ?, ?)',
        [name, price, description, category, estimatedDelivery]
    )
    const productId = result.insertId;
    if (images && images.length > 0) {
      const imagePromises = images.map(async (image) => {
        const [imageResult] = await db.query(
          'INSERT INTO product_images (product_id, url, id, colour, color_name) VALUES (?, ?, ?, ?, ?)',
          [productId, image.url, image.id, image.color.hex, image.color.name]
        );
        return imageResult.insertId;
      });
      await Promise.all(imagePromises);
    }
    res.status(201).json({ 
      success: true, 
      message: 'Product created successfully', 
      productId: productId 
    });
    } catch (error) {
    console.error('Error creating product:', error);
    res.status(500).json({
        success: false,
        error: 'Failed to create product. Please try again later.'
    });
  }
});

app.get('/products', async (req, res) => {
  try {
    const [rows] = await db.query('SELECT * FROM products');
    const [images] = await db.query('SELECT * FROM product_images where product_id IN (?)', [rows.map(product => product.id)]);
    const products = rows.map(product => {
      return {
        ...product,
        colors: images.map(image => ({
          url: image.url,
          id: image.id,
          hex: image.colour,
          name: image.color_name

        }))
    }});
    console
    res.json({
      success: true,
      products: products
    });
  } catch (error) {
    console.error('Error fetching products:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch products. Please try again later.'
    });
  }
});

// Main color extraction endpoint with Cloudinary upload
app.post('/upload-image', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'No image file provided'
      });
    }

    const imagePath = req.file.path;
    const removeBackground = req.body.removeBackground !== 'false';

    console.log(`Processing image: ${imagePath}`);
    console.log(`Remove background: ${removeBackground}`);

    const startTime = Date.now();

    // Process image (color extraction + Cloudinary upload simultaneously)
    const result = await processImage(imagePath, req.file.originalname, removeBackground);

    const processingTime = Date.now() - startTime;

    // Clean up uploaded file
    fs.unlink(imagePath, (err) => {
      if (err) console.error('Error deleting temp file:', err);
    });

    // Prepare response
    const response = {
      success: true,
      colors: result.colors,
      image: {
        cloudinary: result.cloudinary,
        original_filename: req.file.originalname,
        file_size: req.file.size,
        processing_time_ms: processingTime,
        remove_background: removeBackground
      },
      metadata: {
        processed_at: new Date().toISOString(),
        processing_time_ms: processingTime
      },
      id: result.cloudinary.public_id,
      url: result.cloudinary.secure_url,
      colorName: result.colors.dominant_color.name,
      colorHex: result.colors.dominant_color.hex,
    };
    console.log(response);

    console.log(`✅ Image processed successfully in ${processingTime}ms`);
    res.json(response);

  } catch (error) {
    console.error('Error processing image:', error);
    
    // Clean up file if it exists
    if (req.file && req.file.path) {
      fs.unlink(req.file.path, (err) => {
        if (err) console.error('Error deleting temp file:', err);
      });
    }

    res.status(500).json({
      success: false,
      error: error.message || 'Internal server error'
    });
  }
});

// Batch processing endpoint with Cloudinary uploads
app.post('/upload-batch', upload.array('images', 10), async (req, res) => {
  try {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'No image files provided'
      });
    }

    const removeBackground = req.body.removeBackground !== 'false';
    const startTime = Date.now();

    console.log(`Processing ${req.files.length} images in batch...`);

    // Process all images simultaneously
    const processingPromises = req.files.map(async (file) => {
      try {
        const result = await processImage(file.path, file.originalname, removeBackground);
        
        // Clean up file
        fs.unlink(file.path, (err) => {
          if (err) console.error('Error deleting temp file:', err);
        });

        return {
          filename: file.originalname,
          success: true,
          colors: result.colors,
          cloudinary: result.cloudinary
        };
      } catch (error) {
        // Clean up file even if processing failed
        fs.unlink(file.path, (err) => {
          if (err) console.error('Error deleting temp file:', err);
        });

        return {
          filename: file.originalname,
          success: false,
          error: error.message
        };
      }
    });

    // Wait for all processing to complete
    const results = await Promise.all(processingPromises);
    const processingTime = Date.now() - startTime;

    const successCount = results.filter(r => r.success).length;
    const failureCount = results.filter(r => !r.success).length;

    console.log(`✅ Batch processing completed: ${successCount} success, ${failureCount} failures in ${processingTime}ms`);

    res.json({
      success: true,
      processed: results.length,
      successful: successCount,
      failed: failureCount,
      processing_time_ms: processingTime,
      results: results
    });

  } catch (error) {
    console.error('Error in batch processing:', error);
    
    // Clean up any uploaded files
    if (req.files) {
      req.files.forEach(file => {
        fs.unlink(file.path, (err) => {
          if (err) console.error('Error deleting temp file:', err);
        });
      });
    }

    res.status(500).json({
      success: false,
      error: error.message || 'Internal server error'
    });
  }
});

// Get color information by name
app.get('/color-info/:colorName', (req, res) => {
  const colorName = req.params.colorName;
  
  // You can extend this with a color database
  const colorInfo = {
    name: colorName,
    category: 'fashion',
    description: `Color information for ${colorName}`,
    timestamp: new Date().toISOString()
  };
  
  res.json(colorInfo);
});

// Get image from Cloudinary by public_id
app.get('/image/:publicId', (req, res) => {
  const publicId = req.params.publicId;
  
  try {
    const imageUrl = cloudinary.url(publicId, {
      transformation: [
        { width: 400, height: 300, crop: 'fit' },
        { quality: 'auto' }
      ]
    });
    
    res.json({
      success: true,
      public_id: publicId,
      url: imageUrl,
      transformations_available: [
        'thumbnail: /image/' + publicId + '/thumbnail',
        'medium: /image/' + publicId + '/medium',
        'large: /image/' + publicId + '/large'
      ]
    });
  } catch (error) {
    res.status(400).json({
      success: false,
      error: 'Invalid public ID or image not found'
    });
  }
});

// Get different sizes of images
app.get('/image/:publicId/:size', (req, res) => {
  const { publicId, size } = req.params;
  
  const sizeConfigs = {
    thumbnail: { width: 150, height: 150, crop: 'fill' },
    medium: { width: 400, height: 300, crop: 'fit' },
    large: { width: 800, height: 600, crop: 'fit' }
  };
  
  const config = sizeConfigs[size] || sizeConfigs.medium;
  
  try {
    const imageUrl = cloudinary.url(publicId, {
      transformation: [config, { quality: 'auto' }]
    });
    
    res.redirect(imageUrl);
  } catch (error) {
    res.status(400).json({
      success: false,
      error: 'Invalid public ID or size parameter'
    });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({
        success: false,
        error: 'File too large. Maximum size is 10MB.'
      });
    }
  }
  
  res.status(500).json({
    success: false,
    error: error.message || 'Internal server error'
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Fashion Color Extractor API running on port ${PORT}`);
  console.log(`📸 Upload endpoint: http://localhost:${PORT}/upload-image`);
  console.log(`📸 Batch upload: http://localhost:${PORT}/upload-batch`);
  console.log(`🔍 Health check: http://localhost:${PORT}/health`);
  console.log(`🌩️ Cloudinary integration: Active`);
});

module.exports = app;