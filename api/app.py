from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
import uuid
import json
from analyzer import analyze_site_plan, create_highlighted_overview
import traceback

app = Flask(__name__, template_folder='templates')
CORS(app)  # Allow requests from React (default localhost:3000)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'site_plan_output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Store the current analysis data globally (in production, use a database)
current_plots_data = []
current_original_image = None

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/analyze', methods=['POST'])
def analyze_image():
    global current_plots_data, current_original_image
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use PNG or JPG'}), 400

        # Get output format, default to 'svg'
        output_format = request.form.get('output_format', 'svg').lower()
        if output_format not in ['svg', 'jpeg']:
            return jsonify({'error': 'Invalid output format. Use SVG or JPEG'}), 400

        # Save uploaded file with unique name
        unique_filename = f"{uuid.uuid4()}.{file.filename.rsplit('.', 1)[1].lower()}"
        image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(image_path)

        # Run analysis
        plots_data = analyze_site_plan(image_path, save_individual_plots=True, output_format=output_format)
        
        # Store data globally for highlighting functionality
        current_plots_data = plots_data
        import cv2
        current_original_image = cv2.imread(image_path)

        # Prepare response
        response = {
            'summary': {
                'total_plots': len(plots_data),
                'plots': []
            }
        }

        # Read JSON data
        json_path = os.path.join(OUTPUT_FOLDER, 'site_plan_data.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                response['summary'] = json.load(f)['site_plan_analysis']

        # Read interactive data
        interactive_data_path = os.path.join(OUTPUT_FOLDER, 'interactive_data.json')
        if os.path.exists(interactive_data_path):
            with open(interactive_data_path, 'r') as f:
                response['interactive_data'] = json.load(f)

        if output_format == 'svg':
            response['svgs'] = {}
            # Read SVG files
            overview_svg = os.path.join(OUTPUT_FOLDER, f'site_plan_overview.{output_format}')
            if os.path.exists(overview_svg):
                with open(overview_svg, 'r') as f:
                    response['svgs']['overview'] = f.read()

            for plot in plots_data:
                # Individual plot SVGs with dimensions
                shape_type = plot['shape_type'].lower()
                area_text = f"{plot['area_value']}"
                svg_filename = f"plot_{plot['id']}_{shape_type}_{area_text}sqm_with_dimensions.{output_format}"
                svg_path = os.path.join(OUTPUT_FOLDER, svg_filename)
                if os.path.exists(svg_path):
                    with open(svg_path, 'r') as f:
                        response['svgs'][f"plot_{plot['id']}"] = f.read()

                # Highlighted overview SVGs
                highlighted_svg_filename = f"plot_{plot['id']}_overview_highlighted.{output_format}"
                highlighted_svg_path = os.path.join(OUTPUT_FOLDER, highlighted_svg_filename)
                if os.path.exists(highlighted_svg_path):
                    with open(highlighted_svg_path, 'r') as f:
                        response['svgs'][f"plot_{plot['id']}_highlighted"] = f.read()
        else:
            # For JPEG and other formats, return base64 encoded images
            import base64
            response['images'] = {}

            overview_image = os.path.join(OUTPUT_FOLDER, f'site_plan_overview.{output_format}')
            if os.path.exists(overview_image):
                with open(overview_image, 'rb') as f:
                    encoded = base64.b64encode(f.read()).decode('utf-8')
                    response['images']['overview'] = f'data:image/{output_format};base64,{encoded}'

            # Read base overview image (without highlights)
            base_overview_image = os.path.join(OUTPUT_FOLDER, f'site_plan_base_overview.{output_format}')
            if os.path.exists(base_overview_image):
                with open(base_overview_image, 'rb') as f:
                    encoded = base64.b64encode(f.read()).decode('utf-8')
                    response['images']['base_overview'] = f'data:image/{output_format};base64,{encoded}'

            for plot in plots_data:
                # Individual plot images with dimensions
                shape_type = plot['shape_type'].lower()
                area_text = f"{plot['area_value']}"
                image_filename = f"plot_{plot['id']}_{shape_type}_{area_text}sqm_with_dimensions.{output_format}"
                image_path = os.path.join(OUTPUT_FOLDER, image_filename)
                if os.path.exists(image_path):
                    with open(image_path, 'rb') as f:
                        encoded = base64.b64encode(f.read()).decode('utf-8')
                        response['images'][f"plot_{plot['id']}"] = f'data:image/{output_format};base64,{encoded}'

                # Highlighted overview images
                highlighted_image_filename = f"plot_{plot['id']}_overview_highlighted.{output_format}"
                highlighted_image_path = os.path.join(OUTPUT_FOLDER, highlighted_image_filename)
                if os.path.exists(highlighted_image_path):
                    with open(highlighted_image_path, 'rb') as f:
                        encoded = base64.b64encode(f.read()).decode('utf-8')
                        response['images'][f"plot_{plot['id']}_highlighted"] = f'data:image/{output_format};base64,{encoded}'

        # Clean up uploaded file
        if os.path.exists(image_path):
            os.remove(image_path)

        return jsonify(response), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f"Analysis failed: {str(e)}"}), 500

@app.route('/highlight-plot', methods=['POST'])
def highlight_plot():
    global current_plots_data, current_original_image
    
    try:
        data = request.get_json()
        if not data or 'plot_id' not in data:
            return jsonify({'error': 'Plot ID is required'}), 400
        
        plot_id = data['plot_id']
        
        # Validate that we have the required data
        if not current_plots_data or current_original_image is None:
            return jsonify({'error': 'No analysis data available. Please run analysis first.'}), 400
        
        # Find the plot
        selected_plot = next((plot for plot in current_plots_data if plot['id'] == plot_id), None)
        if not selected_plot:
            return jsonify({'error': f'Plot {plot_id} not found'}), 404
        
        # Generate highlighted overview
        highlighted_svg = create_highlighted_overview(current_original_image, current_plots_data, plot_id)
        
        if highlighted_svg:
            return jsonify({'highlighted_overview': highlighted_svg}), 200
        else:
            return jsonify({'error': 'Failed to generate highlighted overview'}), 500
            
    except Exception as e:
        print(f"Error highlighting plot: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f"Highlighting failed: {str(e)}"}), 500

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Site Analyzer API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            h1 { color: #2c3e50; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; }
            .method { font-weight: bold; color: #007bff; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏗️ Site Analyzer API</h1>
            <p>A Flask-based service for analyzing site plans using computer vision techniques.</p>

            <h2>Available Endpoints</h2>

            <div class="endpoint">
                <span class="method">GET</span> /<br>
                <small>This homepage with service information</small>
            </div>

            <div class="endpoint">
                <span class="method">GET</span> /info<br>
                <small>Detailed service information and API documentation</small>
            </div>

            <div class="endpoint">
                <span class="method">POST</span> /analyze<br>
                <small>Upload and analyze a site plan image</small>
            </div>

            <div class="endpoint">
                <span class="method">POST</span> /highlight-plot<br>
                <small>Generate highlighted overview for a specific plot</small>
            </div>

            <div class="endpoint">
                <span class="method">GET</span> /site_plan_output/<filename><br>
                <small>Retrieve generated analysis output files</small>
            </div>

            <h2>Quick Start</h2>
            <p>Visit <a href="/info">/info</a> for detailed API documentation and usage examples.</p>

            <h2>Frontend</h2>
            <p>The companion frontend application is available in the sister directory. See the frontend README for setup instructions.</p>
        </div>
    </body>
    </html>
    '''

@app.route('/info')
def info():
    try:
        return render_template('info.html')
    except:
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Site Analyzer API - Info</title>
        </head>
        <body>
            <h1>Site Analyzer API</h1>
            <p>Service information template not found. Please check the templates directory.</p>
        </body>
        </html>
        '''

@app.route('/site_plan_output/<path:filename>')
def serve_output_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
