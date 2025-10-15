import cv2
import numpy as np
import matplotlib
matplotlib.use('SVG')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MPLPolygon
import io
import base64
import os
import json
import re

# Optional imports
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    print("Warning: pytesseract not found. OCR text extraction will be disabled.")


def load_and_preprocess_image(image_path):
    """
    Advanced preprocessing for clean line extraction
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not load image from {image_path}")

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    binary = cv2.adaptiveThreshold(
        cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 2
    )
    kernel_clean = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)

    return original_image, binary


def detect_scale_and_convert_units_accurate(image):
    """
    More accurate scale detection for proper meter conversion
    """
    scale_factor = 0.05
    detected_scale = False
    scale_info = "estimated"

    if HAS_TESSERACT:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            regions = [
                gray[int(h*0.8):h, 0:int(w*0.3)],
                gray[int(h*0.8):h, int(w*0.7):w],
                gray[0:int(h*0.2), 0:int(w*0.3)],
                gray[0:int(h*0.2), int(w*0.7):w]
            ]
            for region in regions:
                if region.size > 0:
                    region_enhanced = cv2.resize(region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    custom_config = r'--oem 3 --psm 6'
                    text = pytesseract.image_to_string(region_enhanced, config=custom_config).lower()
                    scale_patterns = [
                        (r'1\s*:\s*(\d+)', lambda x: 1/int(x)),
                        (r'(\d+)\s*mm\s*=\s*(\d+\.?\d*)\s*m', lambda x, y: float(y)/(float(x)/1000)),
                        (r'(\d+)\s*cm\s*=\s*(\d+\.?\d*)\s*m', lambda x, y: float(y)/(float(x)/100)),
                        (r'(\d+\.?\d*)\s*m', lambda x: float(x)/100),
                        (r'scale.*?1\s*:\s*(\d+)', lambda x: 1/int(x))
                    ]
                    for pattern, converter in scale_patterns:
                        matches = re.findall(pattern, text)
                        if matches:
                            try:
                                if isinstance(matches[0], tuple):
                                    scale_factor = converter(*matches[0])
                                else:
                                    scale_factor = converter(matches[0])
                                detected_scale = True
                                scale_info = f"detected from drawing: {matches[0]}"
                                print(f"Found scale: {matches[0]} -> {scale_factor:.4f} m/pixel")
                                break
                            except:
                                continue
                        if detected_scale:
                            break
        except Exception as e:
            print(f"Scale detection failed: {e}")

    if not detected_scale:
        print("No scale detected. Using estimation: 20 pixels ≈ 1 meter")
        scale_info = "estimated (20px = 1m)"

    return scale_factor, detected_scale, scale_info


def get_shape_type(num_sides):
    """
    Get descriptive name for polygon based on number of sides (4+ sides only)
    """
    shape_names = {
        4: "Quadrilateral", 5: "Pentagon", 6: "Hexagon",
        7: "Heptagon", 8: "Octagon"
    }
    return shape_names.get(num_sides, f"{num_sides}-sided polygon")


def detect_all_polygon_plots(binary_image, original_image):
    """
    Detect all types of polygon plots (quadrilateral, pentagonal, hexagonal, etc.) - excluding triangles
    """
    print("Detecting all polygon plots (4+ sides)...")
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygon_plots = []
    min_area = 3000
    max_area = original_image.shape[0] * original_image.shape[1] * 0.4

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            epsilon = 0.015 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                perimeter = cv2.arcLength(contour, True)
                perimeter_ratio = perimeter / np.sqrt(area) if area > 0 else 0
                if solidity > 0.3 and perimeter_ratio < 20:
                    M = cv2.moments(contour)
                    centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (x + w//2, y + h//2)
                    polygon_plots.append({
                        'contour': contour,
                        'approx': approx,
                        'area': area,
                        'centroid': centroid,
                        'bounding_rect': (x, y, w, h),
                        'num_sides': len(approx),
                        'solidity': solidity,
                        'perimeter_ratio': perimeter_ratio,
                        'shape_type': get_shape_type(len(approx))
                    })

    polygon_plots.sort(key=lambda x: x['area'], reverse=True)
    print(f"Found {len(polygon_plots)} polygon plots")
    for i, plot in enumerate(polygon_plots):
        print(f"  Plot {i+1}: {plot['shape_type']} ({plot['num_sides']} sides), Area: {plot['area']:.0f} pixels")
    return polygon_plots


def detect_rectangular_plots_alternative(binary_image, original_image):
    """
    Alternative method for detecting rectangular plots when primary method fails
    """
    print("Using alternative rectangular plot detection...")
    
    # Apply morphological operations to enhance rectangular shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary_enhanced = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    binary_enhanced = cv2.morphologyEx(binary_enhanced, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(binary_enhanced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangular_plots = []
    min_area = 2000
    max_area = original_image.shape[0] * original_image.shape[1] * 0.5
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Check if it's roughly rectangular
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            extent = float(area) / rect_area
            
            if extent > 0.7:  # At least 70% of bounding rectangle
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 4:  # At least 4 vertices
                    M = cv2.moments(contour)
                    centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (x + w//2, y + h//2)
                    
                    rectangular_plots.append({
                        'contour': contour,
                        'approx': approx,
                        'area': area,
                        'centroid': centroid,
                        'bounding_rect': (x, y, w, h),
                        'num_sides': len(approx),
                        'shape_type': get_shape_type(len(approx))
                    })
    
    rectangular_plots.sort(key=lambda x: x['area'], reverse=True)
    print(f"Alternative method found {len(rectangular_plots)} plots")
    return rectangular_plots


def calculate_polygon_edge_lengths(vertices, scale_factor):
    """
    Calculate the length of each edge of a polygon in meters
    """
    edge_lengths = []
    n_vertices = len(vertices)
    for i in range(n_vertices):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n_vertices]
        pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        meter_distance = pixel_distance * scale_factor
        edge_lengths.append({
            'edge_index': i,
            'start_point': p1.tolist(),
            'end_point': p2.tolist(),
            'length_pixels': pixel_distance,
            'length_meters': round(meter_distance, 1)
        })
    return edge_lengths


def detect_dimension_labels_enhanced(image, plot_contour, plot_bounds, edge_lengths):
    """
    Optimized dimension detection to match OCR text with calculated edge lengths
    """
    x, y, w, h = plot_bounds
    margin = 100
    search_x = max(0, x - margin)
    search_y = max(0, y - margin)
    search_w = min(image.shape[1] - search_x, w + 2*margin)
    search_h = min(image.shape[0] - search_y, h + 2*margin)
    detected_dimensions = []

    if HAS_TESSERACT and search_w > 0 and search_h > 0:
        try:
            search_roi = image[search_y:search_y+search_h, search_x:search_x+search_w]
            roi_gray = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY) if len(search_roi.shape) == 3 else search_roi
            roi_enhanced = cv2.resize(roi_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.'
            ocr_data = pytesseract.image_to_data(roi_enhanced, config=custom_config,
                                               output_type=pytesseract.Output.DICT)
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                if text and text.replace('.', '').isdigit():
                    conf = int(ocr_data['conf'][i])
                    if conf > 50:
                        scaled_x = int(ocr_data['left'][i] / 2) + search_x
                        scaled_y = int(ocr_data['top'][i] / 2) + search_y
                        value = float(text)
                        if 1 <= value <= 200:
                            detected_dimensions.append({
                                'value': value,
                                'x': scaled_x,
                                'y': scaled_y,
                                'confidence': conf
                            })
        except Exception as e:
            print(f"Dimension detection failed: {e}")

    final_dimensions = []
    for i, edge in enumerate(edge_lengths):
        edge_info = {
            'edge_index': i,
            'start_point': edge['start_point'],
            'end_point': edge['end_point'],
            'length_pixels': edge['length_pixels'],
            'length_meters': edge['length_meters'],
            'source': 'calculated',
            'confidence': 100
        }
        if detected_dimensions:
            edge_mid_x = (edge['start_point'][0] + edge['end_point'][0]) / 2
            edge_mid_y = (edge['start_point'][1] + edge['end_point'][1]) / 2
            best_match = None
            min_distance = float('inf')
            for detection in detected_dimensions:
                distance = np.sqrt((detection['x'] - edge_mid_x)**2 + (detection['y'] - edge_mid_y)**2)
                value_diff = abs(detection['value'] - edge['length_meters'])
                if distance < 80 and value_diff < edge['length_meters'] * 0.3:
                    if distance < min_distance:
                        min_distance = distance
                        best_match = detection
            if best_match and best_match['confidence'] > 60:
                edge_info.update({
                    'length_meters': best_match['value'],
                    'source': 'OCR_detected',
                    'confidence': best_match['confidence'],
                    'detection_distance': min_distance
                })
        final_dimensions.append(edge_info)
    return final_dimensions


def extract_area_from_plot_center(image, plot_bounds):
    """
    Extract the area value specifically from the center of the plot
    """
    x, y, w, h = plot_bounds
    center_margin = 0.3
    center_x = x + int(w * (0.5 - center_margin/2))
    center_y = y + int(h * (0.5 - center_margin/2))
    center_w = int(w * center_margin)
    center_h = int(h * center_margin)
    
    # Ensure we don't go outside image boundaries
    center_x = max(0, center_x)
    center_y = max(0, center_y)
    center_w = min(center_w, image.shape[1] - center_x)
    center_h = min(center_h, image.shape[0] - center_y)
    
    if center_w <= 0 or center_h <= 0:
        # Fallback to estimated area if ROI is invalid
        pixel_area = plot_bounds[2] * plot_bounds[3]
        estimated_area = round(pixel_area * 0.0025)  # Rough estimate: 0.0025 sq.m per pixel
        return estimated_area
    
    center_roi = image[center_y:center_y+center_h, center_x:center_x+center_w]

    if HAS_TESSERACT and center_roi.size > 0:
        try:
            roi_gray = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY) if len(center_roi.shape) == 3 else center_roi
            roi_enhanced = cv2.resize(roi_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.'
            text = pytesseract.image_to_string(roi_enhanced, config=custom_config).strip()
            
            numbers = re.findall(r'\d+\.?\d*', text)
            if numbers:
                # Filter for reasonable area values (assuming residential plots)
                areas = [float(num) for num in numbers if 50 <= float(num) <= 2000]
                if areas:
                    return max(areas)
                else:
                    # If no reasonable areas found, return the largest number
                    all_nums = [float(num) for num in numbers]
                    if all_nums:
                        return max(all_nums)
        except Exception as e:
            print(f"Area extraction failed: {e}")

    # Fallback to estimated area based on pixel area
    pixel_area = plot_bounds[2] * plot_bounds[3]
    estimated_area = round(pixel_area * 0.0025)  # Rough estimate: 0.0025 sq.m per pixel
    return estimated_area


def create_highlighted_overview(original_image, plots_data, selected_plot_id, output_format='svg', show_labels=False):
    """
    Create a dynamically highlighted overview where only the selected plot is highlighted
    with a distinct color while others remain in default colors
    """
    print(f"Creating highlighted overview for plot {selected_plot_id}...")
    
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        ax.imshow(original_rgb, alpha=0.7)

        # Define colors for highlighting with transparency
        selected_colors = {
            'fill': '#f7eb07cc',  # Yellow with 80% opacity
            'stroke': '#262323'   # Dark stroke
        }
        
        default_colors = {
            'fill': "#ffffff", 
            'stroke': '#f7eb07'   # Yellow stroke
        }

        for plot in plots_data:
            vertices = plot['approx'].reshape(-1, 2)
            
            # Use different colors based on selection
            if plot['id'] == selected_plot_id:
                # Highlight the selected plot
                polygon = MPLPolygon(vertices,
                                   facecolor=selected_colors['fill'],
                                   edgecolor=selected_colors['stroke'],
                                   linewidth=3)
            else:
                # Use default colors for other plots
                polygon = MPLPolygon(vertices,
                                   facecolor=default_colors['fill'],
                                   edgecolor=default_colors['stroke'],
                                   linewidth=1)
            
            # Set GID for interactive functionality
            polygon.set_gid(f"plot_{plot['id']}")
            ax.add_patch(polygon)
            
            # Display area value instead of plot ID
            if show_labels:
                centroid = plot['centroid']
                text_color = 'white' if plot['id'] == selected_plot_id else 'black'
                font_weight = 'bold' if plot['id'] == selected_plot_id else 'normal'
                font_size = 14 if plot['id'] == selected_plot_id else 12

                # Show area value at centroid
                # area_text = f"{plot['area_value']}"

                # ax.text(centroid[0], centroid[1], area_text,
                #         fontsize=font_size, fontweight=font_weight, ha='center', va='center',
                #         color=text_color,
                #         bbox=dict(facecolor='white' if plot['id'] != selected_plot_id else 'black',
                #                  alpha=0.8, pad=3, edgecolor='none'))

            # Add edge dimensions for highlighted plot
            if plot['id'] == selected_plot_id:
                for dim in plot.get('edge_dimensions', []):
                    start = dim['start_point']
                    end = dim['end_point']
                    mid_x = (start[0] + end[0]) / 2
                    mid_y = (start[1] + end[1]) / 2

                    # Add dimension text in meters
                    # ax.text(mid_x, mid_y, f"{dim['length_meters']}m",
                    #         ha='center', va='center', fontsize=10, fontweight='bold',
                    #         color='black', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.9))

        # ax.set_title(f'Site Plan - Plot {selected_plot_id} Highlighted',
        #             fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()

        # Convert to SVG string
        if output_format == 'svg':
            svg_buffer = io.StringIO()
            plt.savefig(svg_buffer, format='svg', bbox_inches='tight')
            svg_content = svg_buffer.getvalue()
            svg_buffer.close()
            plt.close()
            return svg_content
        else:
            # For other formats, convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format=output_format, bbox_inches='tight', dpi=300)
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            img_buffer.close()
            plt.close()
            return f'data:image/{output_format};base64,{img_base64}'
            
    except Exception as e:
        print(f"Error creating highlighted overview: {e}")
        plt.close()
        return None


def create_individual_highlighted_overviews(original_image, plots_data, output_dir='site_plan_output', output_format='svg', show_labels=False):
    """
    Create individual overview images with each plot highlighted one at a time
    """
    print("Creating individual highlighted overview images...")

    for plot in plots_data:
        highlighted_svg = create_highlighted_overview(original_image, plots_data, plot['id'], output_format, show_labels)
        if highlighted_svg:
            filename = f'plot_{plot["id"]}_overview_highlighted.{output_format}'
            filepath = os.path.join(output_dir, filename)

            if output_format == 'svg':
                with open(filepath, 'w') as f:
                    f.write(highlighted_svg)
            else:
                # For other formats, the function returns base64 data
                img_data = base64.b64decode(highlighted_svg.split(',')[1])
                with open(filepath, 'wb') as f:
                    f.write(img_data)

            print(f"Saved: {filename}")


def create_individual_plot_images_enhanced(plots_data, output_dir='site_plan_output', save_individual_plots=True, output_format='svg'):
    """
    Create individual plot images with enhanced details
    """
    if not save_individual_plots:
        return
    
    print("Creating individual plot images...")
    
    for plot in plots_data:
        try:
            create_single_plot_image(plot, output_dir, output_format)
        except Exception as e:
            print(f"Failed to create image for plot {plot['id']}: {e}")


def create_single_plot_image(plot_data, output_dir, output_format='svg'):
    """
    Create a detailed image for a single plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the vertices
    vertices = np.array(plot_data['vertices'])
    vertices = np.vstack([vertices, vertices[0]])  # Close the polygon
    
    ax.plot(vertices[:, 0], vertices[:, 1], 'b-', linewidth=2, marker='o', markersize=6)
    ax.fill(vertices[:, 0], vertices[:, 1], alpha=0.3, color='lightblue')
    
    # Add vertex labels
    for i, (x, y) in enumerate(plot_data['vertices']):
        ax.annotate(f'V{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    # Add edge dimensions
    for dim in plot_data.get('edge_dimensions', []):
        start = dim['start_point']
        end = dim['end_point']
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        # Add dimension text
        source_indicator = "📖" if dim.get('source') == 'OCR_detected' else "📏"
        ax.text(mid_x, mid_y, f"{source_indicator}{dim['length_meters']}m",
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Add plot info with area value prominently displayed
    info_text = f"Plot {plot_data['id']} ({plot_data['shape_type']})\n"
    info_text += f"Area: {plot_data['area_value']} sq.m\n"
    info_text += f"Vertices: {len(plot_data['vertices'])}"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add area value at centroid
    centroid_x = np.mean([v[0] for v in plot_data['vertices']])
    centroid_y = np.mean([v[1] for v in plot_data['vertices']])
    ax.text(centroid_x, centroid_y, f"{plot_data['area_value']}", 
            fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9))
    
    ax.set_title(f'Plot {plot_data["id"]} - Detailed View', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    plt.tight_layout()
    
    filename = f'plot_{plot_data["id"]}_detail.{output_format}'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, format=output_format, bbox_inches='tight', dpi=300 if output_format == 'jpeg' else None)
    plt.close()
    
    print(f"Saved: {filename}")


def create_styled_output(original_image, plots_data, output_dir='site_plan_output', save_individual_plots=True, output_format='svg', interactive_mode=True, highlighted_plot_id=None, show_labels=False):
    """
    Updated create_styled_output to support dynamic highlighting and display area values
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Creating styled visualization...")
    fig, ax = plt.subplots(figsize=(12, 8))
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    ax.imshow(original_rgb, alpha=0.7)

    # Define colors
    selected_colors = {
        'fill': '#f7eb07cc',    # Bright yellow for highlighted plot (80% opacity)
        'stroke': "#262323"   # Dark stroke
    }
    
    default_colors = {
        'fill': "#ffffff",    # Default white
        'stroke': '#f7eb07'   # Default yellow stroke
    }

    for plot in plots_data:
        vertices = plot['approx'].reshape(-1, 2)
        
        # Choose colors based on whether this plot is highlighted
        if highlighted_plot_id and plot['id'] == highlighted_plot_id:
            colors = selected_colors
            linewidth = 3
        else:
            colors = default_colors
            linewidth = 1

        if interactive_mode:
            polygon = MPLPolygon(vertices, facecolor=colors['fill'], edgecolor=colors['stroke'], linewidth=linewidth)
            polygon.set_gid(f"plot_{plot['id']}")
        else:
            polygon = MPLPolygon(vertices, facecolor=colors['fill'], edgecolor=colors['stroke'], linewidth=linewidth)

        ax.add_patch(polygon)
        centroid = plot['centroid']
        
        # Adjust text styling for highlighted plot and display area value
        text_color = 'white' if highlighted_plot_id and plot['id'] == highlighted_plot_id else 'black'
        font_weight = 'bold' if highlighted_plot_id and plot['id'] == highlighted_plot_id else 'normal'
        font_size = 14 if highlighted_plot_id and plot['id'] == highlighted_plot_id else 12
        
        # Display area value instead of plot ID
        area_text = f"{plot['area_value']}"
        
        ax.text(centroid[0], centroid[1], area_text,
                fontsize=font_size, fontweight=font_weight, ha='center', va='center',
                color=text_color,
                bbox=dict(facecolor='white' if not (highlighted_plot_id and plot['id'] == highlighted_plot_id) else 'black',
                         alpha=0.8, pad=3, edgecolor='none'))

    title = f'Site Plan - Plot {highlighted_plot_id} Highlighted' if highlighted_plot_id else 'Site Plan - Plot Detection'
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    # Save with appropriate filename
    if highlighted_plot_id:
        overview_filename = f'site_plan_overview_highlighted_{highlighted_plot_id}.{output_format}'
    else:
        overview_filename = f'site_plan_overview.{output_format}' if interactive_mode else f'site_plan_overview_static.{output_format}'
    
    plt.savefig(os.path.join(output_dir, overview_filename), format=output_format, bbox_inches='tight', dpi=300 if output_format == 'jpeg' else None)
    plt.close()

    # Create individual highlighted overview images
    create_individual_highlighted_overviews(original_image, plots_data, output_dir, output_format)

    create_individual_plot_images_enhanced(plots_data, output_dir, save_individual_plots, output_format)
    return True


def save_json_data(plots_data, output_dir='site_plan_output'):
    """
    Save plot data to JSON file for further analysis
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    json_output = {
        'site_plan_analysis': {
            'total_plots': len(plots_data),
            'plots': []
        }
    }
    for plot in plots_data:
        plot_json = {
            'plot_id': plot['id'],
            'vertices': plot['vertices'],
            'centroid': plot['centroid'],
            'area_pixels': int(plot['area_pixels']),
            'area_value': plot['area_value'],
            'edge_dimensions': plot['edge_dimensions'],
            'shape_type': plot['shape_type'],
            'num_sides': plot['num_sides']
        }
        json_output['site_plan_analysis']['plots'].append(plot_json)
    
    json_filepath = os.path.join(output_dir, 'site_plan_data.json')
    with open(json_filepath, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"Saved analysis data to: {json_filepath}")
    return json_filepath


def analyze_site_plan(image_path, save_individual_plots=True, output_format='svg'):
    """
    Main function to analyze site plan and generate output in specified format
    """
    print("Starting site plan analysis...")
    original_image, binary_image = load_and_preprocess_image(image_path)
    print(f"Image loaded: {original_image.shape}")

    scale_factor, detected_scale, scale_info = detect_scale_and_convert_units_accurate(original_image)
    print(f"Scale factor: {scale_factor} meters/pixel ({scale_info})")

    plots = detect_all_polygon_plots(binary_image, original_image)

    if len(plots) == 0:
        print("No plots detected. Trying alternative detection...")
        plots = detect_rectangular_plots_alternative(binary_image, original_image)

    plots_data = []
    for i, plot in enumerate(plots):
        area_value = extract_area_from_plot_center(original_image, plot['bounding_rect'])
        edge_lengths = calculate_polygon_edge_lengths(plot['approx'].reshape(-1, 2), scale_factor)
        edge_dimensions = detect_dimension_labels_enhanced(original_image, plot['contour'],
                                                         plot['bounding_rect'], edge_lengths)
        plot_info = {
            'id': i + 1,
            'contour': plot['contour'],
            'approx': plot['approx'],
            'centroid': plot['centroid'],
            'area_pixels': plot['area'],
            'area_value': area_value,
            'bounding_rect': plot['bounding_rect'],
            'vertices': plot['approx'].reshape(-1, 2).tolist(),
            'edge_dimensions': edge_dimensions,
            'scale_factor': scale_factor,
            'shape_type': plot['shape_type'],
            'num_sides': plot['num_sides']
        }
        plots_data.append(plot_info)

    if plots_data:
        create_styled_output(original_image, plots_data, save_individual_plots=save_individual_plots, output_format=output_format, interactive_mode=(output_format=='svg'))
        save_json_data(plots_data)

    print(f"\n=== SITE PLAN ANALYSIS SUMMARY ===")
    print(f"Scale: {scale_factor} meters/pixel ({scale_info})")
    print(f"Total plots detected: {len(plots_data)}")
    print()
    for plot in plots_data:
        print(f"Plot {plot['id']} ({plot['shape_type']}):")
        print(f"   Area: {plot['area_value']} sq.m (extracted from plot center)")
        for dim in plot.get('edge_dimensions', []):
            status = "detected from drawing" if dim.get('source') == 'OCR_detected' else "calculated"
            print(f"   Edge {dim['edge_index']+1}: {dim['length_meters']} m ({status})")
        pixel_area_sqm = plot['area_pixels'] * (scale_factor ** 2)
        print(f"   Calculated area from dimensions: {pixel_area_sqm:.1f} sq.m")
        print()

    return plots_data


# Utility functions for interactive usage
def highlight_plot(image_path, plot_id, output_dir='site_plan_output'):
    """
    Convenience function to highlight a specific plot
    """
    plots_data = analyze_site_plan(image_path, save_individual_plots=False, output_format='svg')
    if plots_data:
        original_image, _ = load_and_preprocess_image(image_path)
        highlighted_overview = create_highlighted_overview(original_image, plots_data, plot_id, 'svg')
        
        if highlighted_overview:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            filename = f'highlighted_plot_{plot_id}.svg'
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w') as f:
                f.write(highlighted_overview)
            print(f"Highlighted plot {plot_id} saved to: {filepath}")
            return filepath
    return None


def get_plot_summary(image_path):
    """
    Get a quick summary of all detected plots
    """
    plots_data = analyze_site_plan(image_path, save_individual_plots=False, output_format='png')
    
    summary = {
        'total_plots': len(plots_data),
        'plots': []
    }
    
    for plot in plots_data:
        plot_summary = {
            'id': plot['id'],
            'shape_type': plot['shape_type'],
            'num_sides': plot['num_sides'],
            'area_sqm': plot['area_value'],
            'vertices': plot['vertices'],
            'edge_lengths_m': [dim['length_meters'] for dim in plot['edge_dimensions']]
        }
        summary['plots'].append(plot_summary)
    
    return summary


def batch_analyze_plans(image_directory, output_base_dir='batch_analysis'):
    """
    Analyze multiple site plans in a directory
    """
    import glob
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_directory, ext)))
        image_files.extend(glob.glob(os.path.join(image_directory, ext.upper())))
    
    results = {}
    
    for image_file in image_files:
        print(f"\n{'='*50}")
        print(f"Analyzing: {os.path.basename(image_file)}")
        print(f"{'='*50}")
        
        try:
            # Create individual output directory for each image
            base_name = os.path.splitext(os.path.basename(image_file))[0]
            output_dir = os.path.join(output_base_dir, base_name)
            
            plots_data = analyze_site_plan(image_file, save_individual_plots=True, output_format='svg')
            results[base_name] = {
                'status': 'success',
                'plots_count': len(plots_data),
                'plots_data': plots_data,
                'output_directory': output_dir
            }
            
        except Exception as e:
            print(f"Error analyzing {image_file}: {e}")
            results[base_name] = {
                'status': 'error',
                'error': str(e)
            }
    
    # Save batch summary
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    summary_file = os.path.join(output_base_dir, 'batch_analysis_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*50}")
    print("BATCH ANALYSIS COMPLETE")
    print(f"{'='*50}")
    print(f"Total files processed: {len(results)}")
    print(f"Successful: {sum(1 for r in results.values() if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results.values() if r['status'] == 'error')}")
    print(f"Summary saved to: {summary_file}")
    
    return results


# Example usage and testing functions
def run_example():
    """
    Example of how to use the site plan analyzer
    """
    # Example usage - replace with your actual image path
    image_path = "site_plan.jpg"
    
    print("Site Plan Analyzer - Example Usage")
    print("=" * 40)
    
    try:
        # Basic analysis
        print("1. Running basic analysis...")
        plots_data = analyze_site_plan(image_path)
        
        # Highlight specific plot
        print("\n2. Highlighting plot 1...")
        highlight_plot(image_path, 1)
        
        # Get summary
        print("\n3. Getting plot summary...")
        summary = get_plot_summary(image_path)
        print(f"Summary: {summary['total_plots']} plots detected")
        
        return plots_data
        
    except Exception as e:
        print(f"Example failed: {e}")
        print("Make sure you have a valid image file at the specified path.")
        return None


if __name__ == "__main__":
    # Run example if script is executed directly
    print("Site Plan Analyzer - Complete Package")
    print("=" * 50)
    print("Available functions:")
    print("- analyze_site_plan(image_path)")
    print("- highlight_plot(image_path, plot_id)")
    print("- get_plot_summary(image_path)")
    print("- batch_analyze_plans(directory_path)")
    print("- run_example()")
    print("=" * 50)
