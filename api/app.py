from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
import uuid
import json  # Added import
from analyzer import analyze_site_plan
import traceback

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'site_plan_output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('info.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
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

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/site_plan_output/<path:filename>')
def serve_output_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
