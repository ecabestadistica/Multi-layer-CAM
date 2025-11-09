import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import tensorflow as tf
import os
import glob

# Use interactive backend
plt.ion()

# ASL Class names
CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]


def analyze_model_architecture(model):
    """Analyze and display model architecture."""
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE ANALYSIS")
    print("="*70)
    
    total_layers = len(model.layers)
    total_conv_layers = 0
    
    print(f"\nTotal layers: {total_layers}")
    print("\nLayer breakdown:")
    print("-" * 70)
    
    backbone_info = None
    backbone_idx = None
    
    for idx, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        layer_name = layer.name
        
        # Count conv layers
        if 'Conv' in layer_type:
            total_conv_layers += 1
        
        # Check if this is a backbone model
        if hasattr(layer, 'layers') and len(layer.layers) > 20:
            num_sublayers = len(layer.layers)
            
            # Count conv layers in backbone
            backbone_conv_count = 0
            for sublayer in layer.layers:
                sublayer_type = type(sublayer).__name__
                if 'Conv' in sublayer_type:
                    backbone_conv_count += 1
            
            total_conv_layers += backbone_conv_count
            
            print(f"\n[{idx}] {layer_name} ({layer_type})")
            print(f"     ├─ Pre-trained backbone with {num_sublayers} internal layers")
            print(f"     ├─ Convolutional layers in backbone: {backbone_conv_count}")
            
            # Try to identify the backbone type
            backbone_type = "Unknown"
            if 'mobilenet' in layer_name.lower():
                backbone_type = "MobileNet"
            elif 'densenet' in layer_name.lower():
                backbone_type = "DenseNet"
            elif 'resnet' in layer_name.lower():
                backbone_type = "ResNet"
            elif 'efficientnet' in layer_name.lower():
                backbone_type = "EfficientNet"
            elif 'vgg' in layer_name.lower():
                backbone_type = "VGG"
            elif 'inception' in layer_name.lower():
                backbone_type = "Inception"
            
            print(f"     └─ Detected type: {backbone_type}")
            backbone_info = (layer_name, backbone_type, num_sublayers)
            backbone_idx = idx
        else:
            # Regular layer
            output_shape = layer.output_shape if hasattr(layer, 'output_shape') else 'N/A'
            print(f"[{idx}] {layer_name:30s} ({layer_type:20s}) -> {output_shape}")
    
    print("-" * 70)
    
    # Summary
    print("\nArchitecture Summary:")
    print(f"  ✓ Total convolutional layers detected: {total_conv_layers}")
    if backbone_info:
        print(f"  ✓ Contains pre-trained backbone: {backbone_info[1]} ({backbone_info[0]})")
        print(f"  ✓ Backbone has {backbone_info[2]} internal layers")
        print(f"  ✓ Backbone at position: {backbone_idx}")
        print(f"  ✓ Pre-backbone layers: {backbone_idx}")
        print(f"  ✓ Post-backbone layers: {total_layers - backbone_idx - 1}")
    else:
        print("  ℹ No pre-trained backbone detected")
        print("  ℹ Model appears to be custom or fully trained")
    
    print("\nInput shape:", model.input_shape)
    print("Output shape:", model.output_shape)
    print("="*70 + "\n")
    
    return backbone_info, backbone_idx


def get_img_array(img_path, size=(192, 192)):
    """Load and preprocess image."""
    img = tf.keras.utils.load_img(img_path, target_size=size)
    array = tf.keras.utils.img_to_array(img) / 255.0
    array = np.expand_dims(array, axis=0)
    return array


def compute_md_cam_weights(num_layers, power=2.0):
    """MD-CAM weights: bias deeper layers. w_i = i^p / sum(j^p)"""
    indices = np.arange(1, num_layers + 1)
    numerators = indices ** power
    denominator = np.sum(numerators)
    weights = numerators / denominator
    return weights


def compute_mf_cam_weights(num_layers, power=2.0):
    """MF-CAM weights: bias shallower layers. w_i = (L-i+1)^p / sum(j^p)"""
    indices = np.arange(1, num_layers + 1)
    reversed_indices = num_layers - indices + 1
    numerators = reversed_indices ** power
    denominator = np.sum(np.arange(1, num_layers + 1) ** power)
    weights = numerators / denominator
    return weights


def compute_gaussian_cam_weights(num_layers, sigma=None):
    """
    Gaussian-CAM weights: bias middle layers using Gaussian distribution.
    w_i = exp(-(l-μ)²/2σ²) / sum(exp(-(m-μ)²/2σ²))
    
    Args:
        num_layers: Number of layers
        sigma: Standard deviation (if None, uses num_layers)
    """
    if sigma is None:
        sigma = num_layers  
    
    # μ is centered at mid-depth (L/2)
    mu = (num_layers + 1) / 2.0
    
    # Layer indices from 1 to L
    indices = np.arange(1, num_layers + 1)
    
    # Compute Gaussian weights
    numerators = np.exp(-((indices - mu) ** 2) / (2 * sigma ** 2))
    denominator = np.sum(numerators)
    weights = numerators / denominator
    
    return weights


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
    else:  # minmax
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            return (cam - cam_min) / (cam_max - cam_min)
        else:
            return np.zeros_like(cam)


def compute_gradcam_for_layer(model, img_array, class_idx, 
                              backbone_layer, backbone_idx, target_layer_name, img_size):
    """
    Compute Grad-CAM for a specific layer using multi-output model approach.
    """
    try:
        # Find target layer in backbone
        target_layer = None
        for layer in backbone_layer.layers:
            if layer.name == target_layer_name:
                target_layer = layer
                break
        
        if target_layer is None:
            return None
        
        # Create multi-output sub-backbone
        sub_backbone = tf.keras.Model(
            inputs=backbone_layer.input,
            outputs=[target_layer.output, backbone_layer.output]
        )
        
        # Build complete multi-output model
        model_input = model.input
        x = model_input
        
        # Pre-backbone layers
        for i in range(backbone_idx):
            x = model.layers[i](x)
        
        # Get both intermediate and full backbone outputs
        intermediate_out, full_backbone_out = sub_backbone(x)
        
        # Post-backbone layers
        x = full_backbone_out
        for i in range(backbone_idx + 1, len(model.layers)):
            x = model.layers[i](x)
        
        final_output = x
        
        # Create multi-output model
        multi_model = tf.keras.Model(
            inputs=model_input,
            outputs=[intermediate_out, final_output]
        )
        
        # Compute Grad-CAM
        img_tensor = tf.constant(img_array, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            intermediate_output, predictions = multi_model(img_tensor, training=False)
            tape.watch(intermediate_output)
            class_score = predictions[:, class_idx]
        
        grads = tape.gradient(class_score, intermediate_output)
        
        if grads is None:
            return None
        
        # Grad-CAM computation
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        target_out_np = intermediate_output[0].numpy()
        pooled_grads_np = pooled_grads.numpy()
        
        for i in range(len(pooled_grads_np)):
            target_out_np[:, :, i] *= pooled_grads_np[i]
        
        heatmap = np.mean(target_out_np, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Normalize and resize
        heatmap_norm = normalize_cam(heatmap, method='percentile')
        heatmap_resized = tf.image.resize(
            heatmap_norm[np.newaxis, :, :, np.newaxis],
            img_size,
            method='bilinear'
        ).numpy()[0, :, :, 0]
        
        return heatmap_resized
        
    except Exception as e:
        print(f"  ✗ Error for {target_layer_name}: {str(e)[:80]}")
        return None


def compute_gradcam_full_backbone(model, img_array, class_idx,
                                 backbone_layer, backbone_idx, img_size):
    """Compute Grad-CAM on full backbone output (original working method)."""
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
    
    # Normalize and resize
    heatmap_norm = normalize_cam(heatmap, method='percentile')
    heatmap_resized = tf.image.resize(
        heatmap_norm[np.newaxis, :, :, np.newaxis],
        img_size,
        method='bilinear'
    ).numpy()[0, :, :, 0]
    
    return heatmap_resized


def compute_multi_layer_cam(model, img_array, class_idx, layer_names,
                           backbone_layer, backbone_idx, method='md-cam',
                           power=2.0, sigma=None, img_size=(192, 192)):
    """
    Compute MD-CAM, MF-CAM, or G-CAM by fusing multiple layer CAMs.
    
    Args:
        method: 'md-cam', 'mf-cam', or 'g-cam'
        power: Power parameter for MD-CAM and MF-CAM
        sigma: Sigma parameter for G-CAM (if None, uses num_layers)
    """
    num_layers = len(layer_names)
    
    # Compute weights based on method
    if method == 'md-cam':
        weights = compute_md_cam_weights(num_layers, power)
        param_str = f"p={power}"
    elif method == 'mf-cam':
        weights = compute_mf_cam_weights(num_layers, power)
        param_str = f"p={power}"
    else:  # g-cam
        weights = compute_gaussian_cam_weights(num_layers, sigma)
        param_str = f"σ={sigma if sigma else num_layers}"
    
    print(f"  {method.upper()} weights ({param_str}): {weights}")
    
    # Compute CAM for each layer
    cams = []
    for i, layer_name in enumerate(layer_names):
        print(f"    Layer {i+1}/{num_layers}: {layer_name[:40]:40s}", end=" ")
        
        cam = compute_gradcam_for_layer(model, img_array, class_idx,
                                       backbone_layer, backbone_idx,
                                       layer_name, img_size)
        
        if cam is not None:
            cams.append(cam)
            print(f"✓")
        else:
            cams.append(np.zeros(img_size))
            print(f"✗ (using zeros)")
    
    # Weighted fusion
    fused_cam = np.zeros(img_size)
    for i, cam in enumerate(cams):
        fused_cam += weights[i] * cam
    
    # Final normalization
    fused_cam = normalize_cam(fused_cam, method='minmax')
    
    return fused_cam, weights


def plot_cam_comparison(image_path, model, class_names, backbone_layer, backbone_idx,
                       img_size=(192, 192), power=2.0, sigma=None):
    """
    Compare Original Grad-CAM, MD-CAM, MF-CAM, and Gaussian-CAM.
    """
    print(f"\n{'='*70}")
    print(f"Computing CAM Methods Comparison")
    print(f"{'='*70}")
    
    # Load image and predict
    img_array = get_img_array(image_path, size=img_size)
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    predicted_label = class_names[predicted_idx]
    confidence = predictions[predicted_idx]
    
    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Prediction: {predicted_label} ({confidence:.1%})")
    
    # Select layers at different depths
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
    
    print(f"\nUsing {len(test_layers)} layers for multi-layer CAMs:")
    for i, name in enumerate(test_layers):
        print(f"  {i+1}. {name}")
    
    # 1. Original Grad-CAM (full backbone)
    print(f"\n[1/4] Computing Original Grad-CAM (full backbone)...")
    original_cam = compute_gradcam_full_backbone(model, img_array, predicted_idx,
                                                 backbone_layer, backbone_idx, img_size)
    if original_cam is None:
        original_cam = np.zeros(img_size)
        print(f"  ✗ Failed")
    else:
        print(f"  ✓ Success")
    
    # 2. MD-CAM
    print(f"\n[2/4] Computing MD-CAM (bias deeper layers)...")
    md_cam, md_weights = compute_multi_layer_cam(
        model, img_array, predicted_idx, test_layers,
        backbone_layer, backbone_idx, method='md-cam',
        power=power, img_size=img_size
    )
    print(f"  ✓ MD-CAM complete")
    
    # 3. MF-CAM
    print(f"\n[3/4] Computing MF-CAM (bias shallower layers)...")
    mf_cam, mf_weights = compute_multi_layer_cam(
        model, img_array, predicted_idx, test_layers,
        backbone_layer, backbone_idx, method='mf-cam',
        power=power, img_size=img_size
    )
    print(f"  ✓ MF-CAM complete")
    
    # 4. Gaussian-CAM
    print(f"\n[4/4] Computing Gaussian-CAM (bias middle layers)...")
    g_cam, g_weights = compute_multi_layer_cam(
        model, img_array, predicted_idx, test_layers,
        backbone_layer, backbone_idx, method='g-cam',
        sigma=sigma, img_size=img_size
    )
    print(f"  ✓ G-CAM complete")
    
    # Create visualization
    print(f"\n{'='*70}")
    print("Creating comparison visualization...")
    print(f"{'='*70}\n")
    
    fig, axes = plt.subplots(4, 3, figsize=(14, 14))
    
    methods = [
        ('Original Grad-CAM\n(Last Conv Layer)', original_cam, 'lightblue', None),
        (f'MD-CAM\n({len(test_layers)} layers, p={power})', md_cam, 'lightcoral', md_weights),
        (f'MF-CAM\n({len(test_layers)} layers, p={power})', mf_cam, 'lightgreen', mf_weights),
        (f'Gaussian-CAM\n({len(test_layers)} layers, σ={sigma if sigma else len(test_layers)})', 
         g_cam, 'lightyellow', g_weights)
    ]
    
    for row, (title, cam, color, weights) in enumerate(methods):
        # Original image
        axes[row, 0].imshow(img_array[0])
        axes[row, 0].set_title('Original Image', fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Heatmap
        heatmap = np.uint8(cm.jet(cam) * 255)[:, :, :3]
        axes[row, 1].imshow(heatmap)
        axes[row, 1].set_title(title, fontsize=11, fontweight='bold')
        axes[row, 1].axis('off')
        
        # Overlay
        axes[row, 2].imshow(img_array[0])
        axes[row, 2].imshow(heatmap, alpha=0.5)
        axes[row, 2].set_title('Overlay', fontsize=11, fontweight='bold')
        axes[row, 2].axis('off')
    
    # Add title with proper spacing
    fig.suptitle(f'CAM Methods Comparison | Prediction: {predicted_label} ({confidence:.1%})',
                fontsize=13, fontweight='bold', y=0.99)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig


def interactive_image_explorer(model, class_names, backbone_layer, backbone_idx,
                               img_size, power=2.0, sigma=None):
    """
    Interactive image explorer with keyboard controls.
    Press 'N' for next image, 'Q' to quit.
    """
    # Find all test images
    test_image_patterns = ['test_images/*.jpg', 'test_images/*.png', 
                          'test_images/*.jpeg', '*.jpg', '*.png', '*.jpeg']
    
    all_images = []
    for pattern in test_image_patterns:
        all_images.extend(glob.glob(pattern))
    
    if not all_images:
        print("No test images found!")
        return
    
    all_images = sorted(set(all_images))
    print(f"\nFound {len(all_images)} test images")
    for i, img in enumerate(all_images):
        print(f"  {i+1}. {img}")
    
    current_idx = 0
    
    def on_key(event):
        nonlocal current_idx
        
        if event.key == 'n' or event.key == 'N':
            current_idx = (current_idx + 1) % len(all_images)
            plt.close('all')
            show_current_image()
        elif event.key == 'q' or event.key == 'Q':
            plt.close('all')
    
    def show_current_image():
        image_path = all_images[current_idx]
        print(f"\n{'='*70}")
        print(f"Showing image {current_idx + 1}/{len(all_images)}")
        print(f"{'='*70}")
        
        fig = plot_cam_comparison(image_path, model, class_names, 
                                 backbone_layer, backbone_idx,
                                 img_size, power, sigma)
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show(block=True)
    
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Controls:")
    print("  - Press 'N' to view next image")
    print("  - Press 'Q' to quit")
    print("="*70)
    
    show_current_image()


# MAIN
if __name__ == "__main__":
    MODEL_PATH = 'model.h5'  # Generalized model name
    POWER = 2.0  # Power parameter for MD-CAM and MF-CAM
    SIGMA = POWER  # Sigma for G-CAM (If None = use num_layers)
    
    print("="*70)
    print("CAM Methods Comparison Tool")
    print("Includes: Grad-CAM, MD-CAM, MF-CAM, and Gaussian-CAM")
    print("="*70)
    
    # Load model
    print(f"\nLoading model from '{MODEL_PATH}'...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"✗ Error: Model file '{MODEL_PATH}' not found!")
        print("\nPlease ensure your model file is named 'model.h5' in the current directory.")
        exit(1)
    
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Get input size
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    img_size = (input_shape[1], input_shape[2])
    
    print(f"✓ Model loaded! Input size: {img_size}")
    
    # Analyze architecture
    backbone_info, backbone_idx = analyze_model_architecture(model)
    
    if backbone_idx is None:
        print("✗ No backbone found! This tool requires a model with a backbone layer.")
        exit(1)
    
    backbone_layer = model.layers[backbone_idx]
    
    # Run interactive image explorer
    interactive_image_explorer(model, CLASS_NAMES, backbone_layer, backbone_idx,
                              img_size, power=POWER, sigma=SIGMA)
    
    print("\n" + "="*70)
    print("✓ Session complete!")
    print("="*70)
