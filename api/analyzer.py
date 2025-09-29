import cv2
import numpy as np
import matplotlib
matplotlib.use('SVG')  # Use SVG backend for better SVG support
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MPLPolygon
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
import json
import os
import re
from collections import defaultdict

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

def get_shape_type(num_sides):
    """
    Get descriptive name for polygon based on number of sides (4+ sides only)
    """
    shape_names = {
        4: "Quadrilateral", 5: "Pentagon", 6: "Hexagon",
        7: "Heptagon", 8: "Octagon"
    }
    return shape_names.get(num_sides, f"{num_sides}-sided polygon")

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
    center_roi = image[center_y:center_y+center_h, center_x:center_x+center_w]

    if HAS_TESSERACT and center_roi.size > 0:
        try:
            roi_gray = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY) if len(center_roi.shape) == 3 else center_roi
            roi_enhanced = cv2.resize(roi_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.'
            text = pytesseract.image_to_string(roi_enhanced, config=custom_config).strip()
            numbers = re.findall(r'\d+\.?\d*', text)
            if numbers:
                areas = [float(num) for num in numbers if float(num) >= 50]
                return max(areas) if areas else max([float(num) for num in numbers])
        except Exception as e:
            print(f"Area extraction failed: {e}")

    pixel_area = plot_bounds[2] * plot_bounds[3]
    estimated_area = round(pixel_area / 1000, 1)
    return estimated_area

def create_styled_output(original_image, plots_data, output_dir='site_plan_output', save_individual_plots=True, output_format='svg'):
    """
    Create styled output with yellow highlighting, saving in specified format
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Creating styled visualization...")
    print(f"About to call create_individual_highlighted_overviews with {len(plots_data)} plots")
    fig, ax = plt.subplots(figsize=(12, 8))
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    ax.imshow(original_rgb, alpha=0.7)

    for i, plot in enumerate(plots_data):
        vertices = plot['approx'].reshape(-1, 2)
        polygon = MPLPolygon(vertices, facecolor='yellow', edgecolor='orange',
                           linewidth=2, alpha=0.6)
        ax.add_patch(polygon)
        centroid = plot['centroid']
        area_text = f"{plot['area_value']}"
        ax.text(centroid[0], centroid[1], area_text,
                fontsize=12, fontweight='bold', ha='center', va='center')

    ax.set_title('Site Plan - Plot Detection', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    overview_filename = f'site_plan_overview.{output_format}'
    plt.savefig(os.path.join(output_dir, overview_filename), format=output_format, bbox_inches='tight', dpi=300 if output_format == 'jpeg' else None)
    plt.close()

    # Create individual highlighted overview images
    create_individual_highlighted_overviews(original_image, plots_data, output_dir, output_format)

    create_individual_plot_images_enhanced(plots_data, output_dir, save_individual_plots, output_format)
    return True

def create_individual_highlighted_overviews(original_image, plots_data, output_dir, output_format='svg'):
    """
    Create individual overview images where only one plot is highlighted with dimension labels
    positioned along the actual edge angles
    """
    print("Creating individual highlighted overview SVGs with dimension labels...")
    print(f"Number of plots: {len(plots_data)}")
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    for plot in plots_data:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(original_rgb, alpha=0.7)

        # Only highlight the current plot
        plot_polygon = Polygon(plot['approx'].reshape(-1, 2))
        plot_centroid = np.array(plot_polygon.centroid.coords[0])

        # Fallback to moments centroid if shapely fails
        if plot_centroid is None:
            plot_centroid = np.array(plot['centroid'])


        vertices = plot['approx'].reshape(-1, 2)
        polygon = MPLPolygon(vertices, facecolor='yellow', edgecolor='orange',
                           linewidth=2, alpha=0.6)
        ax.add_patch(polygon)

        # Add dimension labels positioned along edge angles
        edge_dimensions = plot.get('edge_dimensions', [])
        for dim in edge_dimensions:
            if dim['length_pixels'] < 20:
                continue
            
            start = np.array(dim['start_point'])
            end = np.array(dim['end_point'])
            mid_point = (start + end) / 2
            edge_vector = end - start
            edge_length = np.linalg.norm(edge_vector)

            if edge_length > 10:
                # Calculate edge angle for proper text rotation
                text_angle = np.arctan2(edge_vector[1], edge_vector[0]) * 180 / np.pi

                # Normalize text_angle to prevent upside-down text
                if text_angle > 90:
                    text_angle -= 180
                elif text_angle < -90:
                    text_angle += 180

                # Create perpendicular offset for label positioning
                perp_vector = np.array([-edge_vector[1], edge_vector[0]])
                if np.linalg.norm(perp_vector) > 0:
                    perp_vector = perp_vector / np.linalg.norm(perp_vector) * 15  # Reduced offset distance

                    # Ensure the label is outside the polygon
                    vec_from_centroid = mid_point - plot_centroid
                    if np.dot(vec_from_centroid, perp_vector) < 0:
                        perp_vector *= -1


                    # Position label closer to the edge
                    label_pos = mid_point + perp_vector

                    dim_text = f"{dim['length_meters']}m"
                    text_color = 'darkgreen' if dim.get('source') == 'OCR_detected' else 'darkblue'

                    # Add text with rotation parallel to the edge
                    ax.text(label_pos[0], label_pos[1], dim_text,
                           fontsize=8, fontweight='normal',  # Reduced font weight
                           ha='center', va='center',
                           color=text_color, alpha=0.9,
                           fontfamily='sans-serif',
                           rotation=text_angle)  # Rotate text parallel to edge

        centroid = plot['centroid']
        area_text = f"{plot['area_value']}"
        ax.text(centroid[0], centroid[1], area_text,
                fontsize=12, fontweight='bold', ha='center', va='center')

        ax.set_title(f'Site Plan - Plot {plot["id"]} Highlighted', fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()

        filename = f'plot_{plot["id"]}_overview_highlighted.{output_format}'
        plt.savefig(os.path.join(output_dir, filename), format=output_format, bbox_inches='tight', dpi=300 if output_format == 'jpeg' else None)
        plt.close()


def create_individual_plot_images_enhanced(plots_data, output_dir, save_individual_plots=True, output_format='svg'):
    """
    Create individual plot images for all polygon types with properly aligned edge labels
    """
    print("Creating individual plot images with accurate meter dimensions...")
    n_plots = len(plots_data)
    if n_plots == 0:
        return
    
    cols = int(np.ceil(np.sqrt(n_plots)))
    rows = int(np.ceil(n_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    
    if n_plots == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = np.array(axes).flatten()
    else:
        axes = axes.flatten()
    
    for i in range(len(axes)):
        if i >= n_plots:
            axes[i].axis('off')
    
    for i, plot in enumerate(plots_data):
        ax = axes[i]
        vertices = plot['approx'].reshape(-1, 2)
        center_x = np.mean(vertices[:, 0])
        center_y = np.mean(vertices[:, 1])
        normalized_vertices = vertices - [center_x, center_y]
        max_extent = np.max(np.abs(normalized_vertices))

        plot_polygon = Polygon(vertices)
        plot_centroid = np.array(plot_polygon.centroid.coords[0])
        # Fallback to moments centroid if shapely fails
        if plot_centroid is None:
            plot_centroid = np.array(plot['centroid'])

        scale = 100 / max_extent if max_extent > 0 else 1
        display_vertices = normalized_vertices * scale + [150, 150]
        
        colors = {
            'Quadrilateral': 'yellow', 'Pentagon': 'lightyellow',
            'Hexagon': 'wheat', 'Heptagon': 'khaki', 'Octagon': 'lemonchiffon'
        }
        shape_type = plot.get('shape_type', 'Polygon')
        color = colors.get(shape_type, 'yellow')
        
        polygon = MPLPolygon(display_vertices, facecolor=color, edgecolor='black', 
                           linewidth=2, alpha=0.8)
        ax.add_patch(polygon)
        
        area_text = f"{plot['area_value']}"
        ax.text(150, 150, area_text, fontsize=10, fontweight='bold',
                ha='center', va='center', fontfamily='sans-serif')
        
        edge_dimensions = plot.get('edge_dimensions', [])
        for dim in edge_dimensions:
            if dim['length_pixels'] < 20:
                continue
                
            start = np.array(dim['start_point'])
            end = np.array(dim['end_point'])
            start_display = (start - [center_x, center_y]) * scale + [150, 150]
            end_display = (end - [center_x, center_y]) * scale + [150, 150]
            mid_point = (start_display + end_display) / 2
            edge_vector = end_display - start_display
            edge_length = np.linalg.norm(edge_vector)

            if edge_length > 10:
                # Calculate edge angle for proper text rotation
                text_angle = np.arctan2(edge_vector[1], edge_vector[0]) * 180 / np.pi

                # Normalize text_angle to prevent upside-down text
                if text_angle > 90:
                    text_angle -= 180
                elif text_angle < -90:
                    text_angle += 180

                perp_vector = np.array([-edge_vector[1], edge_vector[0]])
                if np.linalg.norm(perp_vector) > 0:
                    perp_vector = perp_vector / np.linalg.norm(perp_vector) * 18  # Reduced offset

                    # Ensure the label is outside the polygon
                    vec_from_centroid = mid_point - (np.array([center_x, center_y]) * scale + [150, 150])
                    if np.dot(vec_from_centroid, perp_vector) < 0:
                        perp_vector *= -1

                    label_pos = mid_point + perp_vector

                    dim_text = f"{dim['length_meters']}m"
                    text_color = 'darkgreen' if dim.get('source') == 'OCR_detected' else 'darkblue'

                    # Add rotated text parallel to the edge
                    ax.text(label_pos[0], label_pos[1], dim_text,
                           fontsize=7, fontweight='normal',  # Reduced font weight and size
                           ha='center', va='center',
                           color=text_color, alpha=0.9,
                           fontfamily='sans-serif',
                           rotation=text_angle)  # Rotate text parallel to edge
                    
                    # Draw dimension lines closer to the edge
                    line_offset = perp_vector * 0.6  # Reduced line offset
                    line_start = start_display + line_offset
                    line_end = end_display + line_offset
                    ax.plot([line_start[0], line_end[0]],
                           [line_start[1], line_end[1]],
                           'k-', linewidth=0.8, alpha=0.7)  # Thinner line
                    ax.plot([line_start[0], start_display[0] + line_offset[0]],
                           [line_start[1], start_display[1] + line_offset[1]],
                           'k-', linewidth=0.8, alpha=0.7)
                    ax.plot([line_end[0], end_display[0] + line_offset[0]],
                           [line_end[1], end_display[1] + line_offset[1]],
                           'k-', linewidth=0.8, alpha=0.7)
        
        ax.set_xlim(30, 270)
        ax.set_ylim(270, 30)
        ax.set_aspect('equal')
        ax.axis('off')
        num_sides = plot.get('num_sides', len(vertices))
        ax.set_title(f'{shape_type} Plot {plot["id"]} ({num_sides} sides)\nArea: {area_text} sq.m', 
                    fontsize=8, fontweight='bold', pad=10, fontfamily='sans-serif')
        border = patches.Rectangle((40, 40), 220, 220, linewidth=2, 
                                 edgecolor='orange', facecolor='none')
        ax.add_patch(border)
        
        if save_individual_plots:
            create_single_plot_image(plot, i, output_dir, color)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'all_plots_with_dimensions.{output_format}'),
               format=output_format, bbox_inches='tight', dpi=300 if output_format == 'jpeg' else None)
    plt.close()


def create_single_plot_image(plot, index, output_dir, color, output_format='svg'):
    """
    Create a single plot image with properly aligned edge labels
    """
    fig_single, ax_single = plt.subplots(figsize=(6, 6))
    vertices = plot['approx'].reshape(-1, 2)
    center_x = np.mean(vertices[:, 0])
    center_y = np.mean(vertices[:, 1])
    normalized_vertices = vertices - [center_x, center_y]
    max_extent = np.max(np.abs(normalized_vertices))
    scale = 120 / max_extent if max_extent > 0 else 1

    plot_polygon = Polygon(vertices)
    plot_centroid = np.array(plot_polygon.centroid.coords[0])
    # Fallback to moments centroid if shapely fails
    if plot_centroid is None:
        plot_centroid = np.array(plot['centroid'])

    display_vertices = normalized_vertices * scale + [200, 200]
    
    polygon_single = MPLPolygon(display_vertices, facecolor=color, 
                              edgecolor='black', linewidth=2, alpha=0.8)
    ax_single.add_patch(polygon_single)
    
    area_text = f"{plot['area_value']}"
    ax_single.text(200, 200, area_text, fontsize=12, fontweight='bold',
                  ha='center', va='center', fontfamily='sans-serif')
    
    edge_dimensions = plot.get('edge_dimensions', [])
    for dim in edge_dimensions:
        if dim['length_pixels'] < 20:
            continue
            
        start = np.array(dim['start_point'])
        end = np.array(dim['end_point'])
        start_display = (start - [center_x, center_y]) * scale + [200, 200]
        end_display = (end - [center_x, center_y]) * scale + [200, 200]
        mid_point = (start_display + end_display) / 2
        edge_vector = end_display - start_display
        edge_length = np.linalg.norm(edge_vector)

        if edge_length > 10:
            # Calculate edge angle for proper text rotation
            text_angle = np.arctan2(edge_vector[1], edge_vector[0]) * 180 / np.pi

            # Normalize text_angle to prevent upside-down text
            if text_angle > 90:
                text_angle -= 180
            elif text_angle < -90:
                text_angle += 180

            perp_vector = np.array([-edge_vector[1], edge_vector[0]])
            if np.linalg.norm(perp_vector) > 0:
                perp_vector = perp_vector / np.linalg.norm(perp_vector) * 20  # Reduced offset

                # Ensure the label is outside the polygon
                vec_from_centroid = mid_point - (np.array([center_x, center_y]) * scale + [200, 200])
                if np.dot(vec_from_centroid, perp_vector) < 0:
                    perp_vector *= -1

                label_pos = mid_point + perp_vector

                dim_text = f"{dim['length_meters']}m"
                text_color = 'darkgreen' if dim.get('source') == 'OCR_detected' else 'darkblue'

                # Add rotated text parallel to the edge
                ax_single.text(label_pos[0], label_pos[1], dim_text,
                              fontsize=9, fontweight='normal',  # Reduced font weight
                              ha='center', va='center',
                              color=text_color,
                              fontfamily='sans-serif',
                              rotation=text_angle)  # Rotate text parallel to edge
                
                # Draw dimension lines closer to edge
                line_offset = perp_vector * 0.7  # Reduced line offset
                line_start = start_display + line_offset
                line_end = end_display + line_offset
                ax_single.plot([line_start[0], line_end[0]],
                              [line_start[1], line_end[1]],
                              'k-', linewidth=1, alpha=0.8)  # Slightly thinner line
                ax_single.plot([line_start[0], start_display[0] + line_offset[0]],
                              [line_start[1], start_display[1] + line_offset[1]],
                              'k-', linewidth=1, alpha=0.8)
                ax_single.plot([line_end[0], end_display[0] + line_offset[0]],
                              [line_end[1], end_display[1] + line_offset[1]],
                              'k-', linewidth=1, alpha=0.8)
    
    ax_single.set_xlim(50, 350)
    ax_single.set_ylim(350, 50)
    ax_single.set_aspect('equal')
    ax_single.axis('off')
    shape_type = plot.get('shape_type', 'Polygon')
    num_sides = plot.get('num_sides', len(vertices))
    ax_single.set_title(f'{shape_type} Plot {plot["id"]}\nArea: {area_text} sq.m', 
                       fontsize=12, fontweight='bold', pad=15, fontfamily='sans-serif')
    border_single = patches.Rectangle((60, 60), 280, 280, linewidth=3, 
                                    edgecolor='darkorange', facecolor='none')
    ax_single.add_patch(border_single)
    
    filename = f'plot_{plot["id"]}_{shape_type.lower()}_{area_text}sqm_with_dimensions.svg'
    plt.savefig(os.path.join(output_dir, filename),
               format='svg', bbox_inches='tight')
    plt.close()

def detect_rectangular_plots_alternative(binary_image, original_image):
    """
    Alternative detection method with more relaxed parameters
    """
    print("Using alternative detection method...")
    edges = cv2.Canny(binary_image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                           minLineLength=50, maxLineGap=10)
    rectangular_plots = []
    if lines is not None:
        line_mask = np.zeros_like(binary_image)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
        contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                centroid = (x + w//2, y + h//2)
                rectangular_plots.append({
                    'contour': contour,
                    'approx': cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True),
                    'area': area,
                    'centroid': centroid,
                    'bounding_rect': (x, y, w, h),
                    'shape_type': 'Quadrilateral',
                    'num_sides': 4
                })
    return rectangular_plots

def save_json_data(plots_data, output_dir='site_plan_output'):
    """
    Save plot data to JSON file including dimensions
    """
    json_data = {
        'site_plan_analysis': {
            'total_plots': len(plots_data),
            'plots': []
        }
    }
    for plot in plots_data:
        plot_info = {
            'plot_id': plot['id'],
            'area_value': plot['area_value'],
            'area_pixels': plot['area_pixels'],
            'centroid': plot['centroid'],
            'bounding_rect': plot['bounding_rect'],
            'vertices': plot['vertices'],
            'edge_dimensions': plot['edge_dimensions'],
            'shape_type': plot['shape_type'],
            'num_sides': plot['num_sides']
        }
        json_data['site_plan_analysis']['plots'].append(plot_info)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'site_plan_data.json'), 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Data saved to {output_dir}/site_plan_data.json")

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
        create_styled_output(original_image, plots_data, save_individual_plots=save_individual_plots, output_format=output_format)
        save_json_data(plots_data)

    print(f"\n=== SITE PLAN ANALYSIS SUMMARY ===")
    print(f"Scale: {scale_factor} meters/pixel ({scale_info})")
    print(f"Total plots detected: {len(plots_data)}")
    print()
    for plot in plots_data:
        print(f"📍 Plot {plot['id']} ({plot['shape_type']}):")
        print(f"   Area: {plot['area_value']} sq.m (number inside the plot)")
        for dim in plot.get('edge_dimensions', []):
            status = "📖 detected from drawing" if dim.get('source') == 'OCR_detected' else "📏 calculated"
            print(f"   Edge {dim['edge_index']+1}: {dim['length_meters']} m ({status})")
        pixel_area_sqm = plot['area_pixels'] * (scale_factor ** 2)
        print(f"   Calculated area from dimensions: {pixel_area_sqm:.1f} sq.m")
        print()

    return plots_data
