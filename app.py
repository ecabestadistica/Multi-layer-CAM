"""
Flask Web Application for Interactive CAM Visualization
Allows browsing through test images with Original Grad-CAM, MD-CAM, and MF-CAM
"""

from flask import Flask, render_template, jsonify, send_from_directory
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import glob
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Configuration
MODEL_PATH = 'model.h5'
IMAGE_FOLDER = 'test_images'
POWER = 2.0

# ASL Class names
CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

# Global variables
model = None
img_size = None
image_paths = []
backbone_layer = None
backbone_idx = None
pretrained_architecture = None


def detect_pretrained_architecture(model):
    """Detect pre-trained model architecture used."""
    architecture_keywords = {
        'mobilenet': ['MobileNet', 'mobilenet'],
        'resnet': ['ResNet', 'resnet'],
        'densenet': ['DenseNet', 'densenet'],
        'efficientnet': ['EfficientNet', 'efficientnet'],
        'vgg': ['VGG', 'vgg'],
        'inception': ['Inception', 'inception'],
        'xception': ['Xception', 'xception'],
        'nasnet': ['NASNet', 'nasnet']
    }
    
    # Check model name first
    model_name = model.name.lower()
    for arch, keywords in architecture_keywords.items():
        for keyword in keywords:
            if keyword.lower() in model_name:
                return arch.upper()
    
    # Check layer names
    for layer in model.layers:
        layer_name = layer.name.lower()
        for arch, keywords in architecture_keywords.items():
            for keyword in keywords:
                if keyword.lower() in layer_name:
                    return arch.upper()
        
        # Check nested layers (for functional models)
        if hasattr(layer, 'layers'):
            for nested_layer in layer.layers:
                nested_name = nested_layer.name.lower()
                for arch, keywords in architecture_keywords.items():
                    for keyword in keywords:
                        if keyword.lower() in nested_name:
                            return arch.upper()
    
    return "Unknown"


def load_model():
    """Load the model and find backbone."""
    global model, img_size, backbone_layer, backbone_idx, pretrained_architecture
    
    print(f"Loading model from '{MODEL_PATH}'...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    img_size = (input_shape[1], input_shape[2])
    
    # Detect pre-trained architecture
    pretrained_architecture = detect_pretrained_architecture(model)
    
    # Find backbone
    for idx, layer in enumerate(model.layers):
        if hasattr(layer, 'layers') and len(layer.layers) > 100:
            backbone_layer = layer
            backbone_idx = idx
            break
    
    print(f"✓ Model loaded! Input size: {img_size}")
    print(f"✓ Pre-trained architecture: {pretrained_architecture}")
    print(f"✓ Backbone: {backbone_layer.name} at index {backbone_idx}")


def load_images():
    """Load all image paths from the test folder."""
    global image_paths
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext)))
    image_paths.sort()
    
    print(f"✓ Found {len(image_paths)} images")


def get_img_array(img_path, size):
    """Load and preprocess image."""
    img = tf.keras.utils.load_img(img_path, target_size=size)
    array = tf.keras.utils.img_to_array(img) / 255.0
    array = np.expand_dims(array, axis=0)
    return array


def extract_label_from_filename(filename):
    """Extract expected label from filename."""
    basename = os.path.basename(filename)
    if '_' in basename:
        label = basename.split('_')[0].upper()
    else:
        label = basename.split('.')[0].upper()
    return label


def normalize_cam(cam, method='percentile'):
    """Normalize CAM to [0, 1] range."""
    if method == 'percentile':
        p2, p98 = np.percentile(cam, [2, 98])
        cam_clipped = np.clip(cam, p2, p98)
        cam_min = cam_clipped.min()
        cam_max = cam_clipped.max()
        if cam_max - cam_min > 1e-8:
            return (cam_clipped - cam_min) / (cam_max - cam_min)
        else:
            return np.zeros_like(cam)
    else:
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            return (cam - cam_min) / (cam_max - cam_min)
        else:
            return np.zeros_like(cam)


def compute_md_cam_weights(num_layers, power=2.0):
    """MD-CAM weights: bias deeper layers."""
    indices = np.arange(1, num_layers + 1)
    numerators = indices ** power
    denominator = np.sum(numerators)
    weights = numerators / denominator
    return weights


def compute_mf_cam_weights(num_layers, power=2.0):
    """MF-CAM weights: bias shallower layers."""
    indices = np.arange(1, num_layers + 1)
    reversed_indices = num_layers - indices + 1
    numerators = reversed_indices ** power
    denominator = np.sum(np.arange(1, num_layers + 1) ** power)
    weights = numerators / denominator
    return weights


def compute_gaussian_cam_weights(num_layers, sigma=1.0):
    """Gaussian-CAM weights: emphasize middle layers with Gaussian distribution."""
    indices = np.arange(num_layers)
    center = (num_layers - 1) / 2  # Center on middle layer (0-indexed)
    weights = np.exp(-((indices - center) ** 2) / (2 * sigma ** 2))
    weights = weights / np.sum(weights)  # Normalize to sum to 1
    return weights


def compute_gradcam_full_backbone(img_array, class_idx):
    """Compute Grad-CAM on full backbone output."""
    img_tensor = tf.constant(img_array, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        x = img_tensor
        for i in range(backbone_idx):
            x = model.layers[i](x, training=False)
        
        backbone_output = backbone_layer(x, training=False)
        tape.watch(backbone_output)
        
        x = backbone_output
        for i in range(backbone_idx + 1, len(model.layers)):
            x = model.layers[i](x, training=False)
        
        predictions = x
        class_score = predictions[:, class_idx]
    
    grads = tape.gradient(class_score, backbone_output)
    
    if grads is None:
        return None
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    backbone_out_np = backbone_output[0].numpy()
    pooled_grads_np = pooled_grads.numpy()
    
    for i in range(len(pooled_grads_np)):
        backbone_out_np[:, :, i] *= pooled_grads_np[i]
    
    heatmap = np.mean(backbone_out_np, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    heatmap_norm = normalize_cam(heatmap, method='percentile')
    heatmap_resized = tf.image.resize(
        heatmap_norm[np.newaxis, :, :, np.newaxis],
        img_size,
        method='bilinear'
    ).numpy()[0, :, :, 0]
    
    return heatmap_resized


def compute_gradcam_for_layer(img_array, class_idx, target_layer_name):
    """Compute Grad-CAM for a specific layer."""
    try:
        target_layer = None
        for layer in backbone_layer.layers:
            if layer.name == target_layer_name:
                target_layer = layer
                break
        
        if target_layer is None:
            return None
        
        sub_backbone = tf.keras.Model(
            inputs=backbone_layer.input,
            outputs=[target_layer.output, backbone_layer.output]
        )
        
        model_input = model.input
        x = model_input
        
        for i in range(backbone_idx):
            x = model.layers[i](x)
        
        intermediate_out, full_backbone_out = sub_backbone(x)
        
        x = full_backbone_out
        for i in range(backbone_idx + 1, len(model.layers)):
            x = model.layers[i](x)
        
        final_output = x
        
        multi_model = tf.keras.Model(
            inputs=model_input,
            outputs=[intermediate_out, final_output]
        )
        
        img_tensor = tf.constant(img_array, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            intermediate_output, predictions = multi_model(img_tensor, training=False)
            tape.watch(intermediate_output)
            class_score = predictions[:, class_idx]
        
        grads = tape.gradient(class_score, intermediate_output)
        
        if grads is None:
            return None
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        target_out_np = intermediate_output[0].numpy()
        pooled_grads_np = pooled_grads.numpy()
        
        for i in range(len(pooled_grads_np)):
            target_out_np[:, :, i] *= pooled_grads_np[i]
        
        heatmap = np.mean(target_out_np, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        heatmap_norm = normalize_cam(heatmap, method='percentile')
        heatmap_resized = tf.image.resize(
            heatmap_norm[np.newaxis, :, :, np.newaxis],
            img_size,
            method='bilinear'
        ).numpy()[0, :, :, 0]
        
        return heatmap_resized
        
    except Exception as e:
        print(f"Error computing CAM for {target_layer_name}: {e}")
        return None


def compute_multi_layer_cam(img_array, class_idx, layer_names, method='md-cam', power=2.0):
    """Compute MD-CAM, MF-CAM, or Gaussian-CAM."""
    num_layers = len(layer_names)
    
    if method == 'md-cam':
        weights = compute_md_cam_weights(num_layers, power)
    elif method == 'gaussian-cam':
        weights = compute_gaussian_cam_weights(num_layers, sigma=power)
    else:  # mf-cam
        weights = compute_mf_cam_weights(num_layers, power)
    
    cams = []
    for layer_name in layer_names:
        cam = compute_gradcam_for_layer(img_array, class_idx, layer_name)
        if cam is not None:
            cams.append(cam)
        else:
            cams.append(np.zeros(img_size))
    
    fused_cam = np.zeros(img_size)
    for i, cam in enumerate(cams):
        fused_cam += weights[i] * cam
    
    fused_cam = normalize_cam(fused_cam, method='minmax')
    
    return fused_cam, weights.tolist()


def cam_to_heatmap(cam, colormap='jet'):
    """Convert CAM array to colored heatmap using matplotlib colormaps."""
    # Get the colormap
    cmap = plt.get_cmap(colormap)
    
    # Apply colormap (returns RGBA)
    colored = cmap(cam)
    
    # Convert to RGB (0-255)
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    
    return rgb


def array_to_base64(img_array):
    """Convert numpy array to base64 string for HTML."""
    # Ensure values are in 0-255 range
    if img_array.max() <= 1.0:
        img_uint8 = (img_array * 255).astype(np.uint8)
    else:
        img_uint8 = img_array.astype(np.uint8)
    
    img = Image.fromarray(img_uint8)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def create_overlay(img_array, cam):
    """Create overlay of image and CAM heatmap."""
    # Convert CAM to RGB heatmap using proper colormap
    heatmap = cam_to_heatmap(cam, colormap='jet')
    
    # Convert original image
    img_uint8 = (img_array[0] * 255).astype(np.uint8)
    
    # Blend
    alpha = 0.4
    overlay = (img_uint8 * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    
    return overlay


def get_selected_layers():
    """Get selected convolutional layers for multi-layer CAM."""
    conv_layers = []
    for layer in backbone_layer.layers:
        layer_type = type(layer).__name__
        if 'Conv' in layer_type and '2D' in layer_type:
            conv_layers.append(layer.name)
    
    test_layers = []
    if len(conv_layers) >= 3:
        early_idx = len(conv_layers) // 10
        test_layers.append(conv_layers[early_idx])
        
        mid_idx = len(conv_layers) * 4 // 10
        test_layers.append(conv_layers[mid_idx])
        
        late_idx = len(conv_layers) * 8 // 10
        test_layers.append(conv_layers[late_idx])
    
    return test_layers


@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html', 
                         total_images=len(image_paths),
                         class_names=CLASS_NAMES)


@app.route('/api/process/<int:index>')
def process_image(index):
    """Process image at given index and return CAM results."""
    if index < 0 or index >= len(image_paths):
        return jsonify({'error': 'Invalid index'}), 400
    
    image_path = image_paths[index]
    filename = os.path.basename(image_path)
    
    # Load and predict
    img_array = get_img_array(image_path, img_size)
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))
    predicted_label = CLASS_NAMES[predicted_idx]
    confidence = float(predictions[predicted_idx])
    
    # Expected label
    expected_label = extract_label_from_filename(filename)
    
    # Find expected label index
    expected_idx = None
    if expected_label in CLASS_NAMES:
        expected_idx = CLASS_NAMES.index(expected_label)
    
    is_correct = (predicted_label == expected_label)
    
    # Get top 5 predictions
    top5_indices = np.argsort(predictions)[-5:][::-1]
    top5 = [{'label': CLASS_NAMES[i], 'confidence': float(predictions[i])} 
            for i in top5_indices]
    
    # Select layers
    test_layers = get_selected_layers()
    
    # Compute CAMs for predicted class
    original_cam_pred = compute_gradcam_full_backbone(img_array, predicted_idx)
    if original_cam_pred is None:
        original_cam_pred = np.zeros(img_size)
    
    md_cam_pred, md_weights = compute_multi_layer_cam(
        img_array, predicted_idx, test_layers, method='md-cam', power=POWER
    )
    
    mf_cam_pred, mf_weights = compute_multi_layer_cam(
        img_array, predicted_idx, test_layers, method='mf-cam', power=POWER
    )
    
    gaussian_cam_pred, gaussian_weights = compute_multi_layer_cam(
        img_array, predicted_idx, test_layers, method='gaussian-cam', power=POWER
    )
    
    # Create heatmaps and overlays for predicted class
    original_heatmap_pred = cam_to_heatmap(original_cam_pred)
    original_overlay_pred = create_overlay(img_array, original_cam_pred)
    
    md_heatmap_pred = cam_to_heatmap(md_cam_pred)
    md_overlay_pred = create_overlay(img_array, md_cam_pred)
    
    mf_heatmap_pred = cam_to_heatmap(mf_cam_pred)
    mf_overlay_pred = create_overlay(img_array, mf_cam_pred)
    
    gaussian_heatmap_pred = cam_to_heatmap(gaussian_cam_pred)
    gaussian_overlay_pred = create_overlay(img_array, gaussian_cam_pred)
    
    # Convert to base64
    original_img_b64 = array_to_base64(img_array[0])
    
    result = {
        'filename': filename,
        'index': index,
        'total': len(image_paths),
        'predicted_label': predicted_label,
        'expected_label': expected_label,
        'is_correct': is_correct,
        'confidence': confidence,
        'top5': top5,
        'layers': test_layers,
        'md_weights': md_weights,
        'mf_weights': mf_weights,
        'gaussian_weights': gaussian_weights,
        'architecture': pretrained_architecture,
        'images': {
            'original': original_img_b64,
            'predicted': {
                'original_heatmap': array_to_base64(original_heatmap_pred),
                'original_overlay': array_to_base64(original_overlay_pred),
                'md_heatmap': array_to_base64(md_heatmap_pred),
                'md_overlay': array_to_base64(md_overlay_pred),
                'mf_heatmap': array_to_base64(mf_heatmap_pred),
                'mf_overlay': array_to_base64(mf_overlay_pred),
                'gaussian_heatmap': array_to_base64(gaussian_heatmap_pred),
                'gaussian_overlay': array_to_base64(gaussian_overlay_pred)
            }
        }
    }
    
    # If prediction is incorrect and expected label exists, compute CAMs for expected class too
    if not is_correct and expected_idx is not None:
        original_cam_exp = compute_gradcam_full_backbone(img_array, expected_idx)
        if original_cam_exp is None:
            original_cam_exp = np.zeros(img_size)
        
        md_cam_exp, _ = compute_multi_layer_cam(
            img_array, expected_idx, test_layers, method='md-cam', power=POWER
        )
        
        mf_cam_exp, _ = compute_multi_layer_cam(
            img_array, expected_idx, test_layers, method='mf-cam', power=POWER
        )
        
        gaussian_cam_exp, _ = compute_multi_layer_cam(
            img_array, expected_idx, test_layers, method='gaussian-cam', power=POWER
        )
        
        # Create heatmaps and overlays for expected class
        original_heatmap_exp = cam_to_heatmap(original_cam_exp)
        original_overlay_exp = create_overlay(img_array, original_cam_exp)
        
        md_heatmap_exp = cam_to_heatmap(md_cam_exp)
        md_overlay_exp = create_overlay(img_array, md_cam_exp)
        
        mf_heatmap_exp = cam_to_heatmap(mf_cam_exp)
        mf_overlay_exp = create_overlay(img_array, mf_cam_exp)
        
        gaussian_heatmap_exp = cam_to_heatmap(gaussian_cam_exp)
        gaussian_overlay_exp = create_overlay(img_array, gaussian_cam_exp)
        
        result['images']['expected'] = {
            'original_heatmap': array_to_base64(original_heatmap_exp),
            'original_overlay': array_to_base64(original_overlay_exp),
            'md_heatmap': array_to_base64(md_heatmap_exp),
            'md_overlay': array_to_base64(md_overlay_exp),
            'mf_heatmap': array_to_base64(mf_heatmap_exp),
            'mf_overlay': array_to_base64(mf_overlay_exp),
            'gaussian_heatmap': array_to_base64(gaussian_heatmap_exp),
            'gaussian_overlay': array_to_base64(gaussian_overlay_exp)
        }
    
    return jsonify(result)


if __name__ == '__main__':
    # Initialize
    load_model()
    load_images()
    
    if len(image_paths) == 0:
        print(f"✗ No images found in {IMAGE_FOLDER}")
        exit(1)
    
    print("\n" + "="*70)
    print("Starting Flask web server...")
    print("Open your browser and go to: http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, use_reloader=False)
