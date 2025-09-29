# Site Analyzer Backend

A Flask-based backend service for analyzing site plans using computer vision techniques. The service processes uploaded images of site plans, detects polygon-shaped plots, extracts dimensions, and returns detailed analysis results.

## Features

- **Plot Detection**: Automatically detects quadrilateral, pentagonal, hexagonal, heptagonal, octagonal, and other polygon-shaped plots
- **Dimension Extraction**: Extracts edge lengths and calculates accurate area measurements
- **Scale Detection**: Automatically detects or estimates scale from drawings (supports OCR and manual detection)
- **Output Formats**: Supports both SVG and JPEG output formats for visualizations
- **REST API**: Provides endpoints for image upload, analysis, and results retrieval

## Prerequisites

- Python 3.7+
- pip (Python package installer)
- Tesseract OCR (optional, for enhanced text extraction from drawings)

## Local Development Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd site-analyzer-backend-main
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python api/app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

- `GET /` - Home page with basic information
- `GET /info` - Detailed service information
- `POST /analyze` - Upload and analyze a site plan image
  - **Parameters:**
    - `file` (required): Image file (PNG, JPG, JPEG)
    - `output_format` (optional): 'svg' or 'jpeg' (default: 'svg')
  - **Response:** JSON with plot analysis data, SVG/JPEG content, and overview
- `GET /site_plan_output/<filename>` - Retrieve generated output files

## Usage

1. Start the backend server following the setup instructions above
2. Send a POST request to `/analyze` with an image file
3. Receive detailed analysis including:
   - Detected plots with shapes and areas
   - Individual plot visualizations with dimensions
   - Overview of entire site plan with highlighting

## Dependencies

- Flask: Web framework
- opencv-python-headless: Computer vision processing
- numpy: Numerical computations
- matplotlib: Plotting and visualization
- shapely: Geometric operations
- pytesseract: OCR text extraction (optional)

## Project Structure

```
site-analyzer-backend-main/
├── api/
│   ├── app.py              # Main Flask application
│   ├── analyzer.py         # Computer vision analysis logic
│   ├── templates/
│   │   └── info.html       # Info page template
│   ├── uploads/            # Uploaded images (created at runtime)
│   └── site_plan_output/   # Generated analysis outputs (created at runtime)
├── requirements.txt        # Python dependencies
├── Procfile               # Heroku deployment configuration
├── package.json           # Node.js configuration for build process
└── README.md              # This file
```

## Deployment

The application includes configuration for Heroku deployment via the `Procfile` and build scripts in `package.json`.

## Docker Support

This application can be containerized using Docker. Create a `Dockerfile` and `docker-compose.yml` as needed for containerized deployments.

## Troubleshooting

- **Import Errors**: Ensure all dependencies are installed in the virtual environment
- **Port Issues**: Default port is 5000; can be changed via `PORT` environment variable
- **OCR Issues**: If pytesseract fails, OCR features will be disabled but analysis will continue
- **Image Processing**: Ensure uploaded images are clear site plan drawings for best results
