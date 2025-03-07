from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO
from annoy import AnnoyIndex
import time
import cv2
import numpy as np
import json
import imagehash

def process_full_mask(mask):
    """
    Process the mask to extract its outer boundary and find rectangular contours.
    :param mask: Binary mask (0 or 255) as a NumPy array.
    :return: Rectangle corners (if found) or None.
    """
    # Morphological closing to ensure the mask is a complete object
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours of the entire mask
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No contours found in the mask. Skipping...")
        return None

    # Find the largest contour (assume it represents the card's shape)
    largest_contour = max(contours, key=cv2.contourArea)

    hull = cv2.convexHull(largest_contour)

    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)  # Tolerance for approximation
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) != 4:
        print(f"Approximated polygon has {len(approx)} sides. Trying to force 4 sides...")

        # Dynamically adjust epsilon to "force" 4 corners
        for epsilon_factor in np.linspace(0.01, 0.1, 10):  # Incrementally adjust tolerance
            epsilon = epsilon_factor * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            if len(approx) == 4:  # Break the loop once we get 4 points
                break

    # If it still doesn't have 4 sides after various adjustments, return None
    if len(approx) != 4:
        print("Unable to approximate a 4-sided polygon. Skipping...")
        return None

    # If a valid 4-sided polygon is detected, return it
    print(f"Detected 4-sided polygon after adjustment.")
    return approx


def detect_edges_and_corners(image_path, model_path="best.pt", device="cpu"):
    # Timing start
    start_time = time.time()

    # Set the target size for downscaling (for faster processing)
    target_size = (640, 640)  # Width, Height for downscaling

    # 1. Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(model_path).to(device)

    # 2. Load the image
    print("Loading image...")
    image_bytes = image_path.read()  # Read the image file into memory
    img_array = np.frombuffer(image_bytes, np.uint8)  # Convert bytes to a NumPy array with uint8 type
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Decode the image using OpenCV

    if img is None:
        print("Error: Image not found at", image_path)
        return None

    # Resize image for processing (downscaled for faster mask detection)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_downscaled = cv2.resize(img_rgb, target_size)

    print("Image loaded. Starting YOLO prediction...")
    # 3. Predict using YOLO to get masks
    yolo_start = time.time()
    results = model.predict(source=img_downscaled, save=False, device=device)
    print("YOLO Prediction Time:", time.time() - yolo_start)

    # 4. Extract all masks
    masks = results[0].masks.data
    result_output = img_downscaled.copy()  # Use the downscaled image for drawing rectangles

    print(f"Number of detected masks: {len(masks)}")
    mask_processing_start = time.time()

    detected_cards = []

    for i, mask in enumerate(masks[:10]):  # Limit to 10 masks for testing
        print(f"Processing mask {i + 1}/{len(masks)}...")
        # 5. Convert mask to binary for OpenCV processing
        mask_np = mask.cpu().numpy().astype(np.uint8) * 255  # Convert to binary form
        print(f"Mask shape: {mask_np.shape}, Unique values: {np.unique(mask_np)}")

        # Use the new process_full_mask function to detect the rectangle
        rectangle = process_full_mask(mask_np)

        if rectangle is not None:
            # Draw the detected rectangle on the downscaled output image
            cv2.polylines(result_output, [rectangle], isClosed=True, color=(0, 255, 0), thickness=3)

            # Scale the rectangle points back to the original image size
            scale_x = img.shape[1] / target_size[0]  # Scaling factor for width
            scale_y = img.shape[0] / target_size[1]  # Scaling factor for height
            rectangle_original = rectangle * np.array([scale_x, scale_y])

            # Perform perspective transformation on the original high-resolution image
            # Define the destination points for the transformation (upright rectangle with correct aspect ratio)
            card_width = 500  # Width of the output card (high resolution)
            card_height = 700  # Height of the output card (matches Pokémon card aspect ratio ~2.5:3.5)
            dst_points = np.array([
                [0, 0],
                [card_width - 1, 0],
                [card_width - 1, card_height - 1],
                [0, card_height - 1]
            ], dtype=np.float32)

            # Ensure rectangle is in the correct order (top-left, top-right, bottom-right, bottom-left)
            rect_points = rectangle_original.reshape(4, 2).astype(np.float32)
            rect_points = sort_rectangle_points(rect_points)

            # Compute the perspective transform matrix
            M = cv2.getPerspectiveTransform(rect_points, dst_points)

            # Apply the perspective transformation to the original high-resolution image
            warped_image = cv2.warpPerspective(img_rgb, M, (card_width, card_height), flags=cv2.INTER_LANCZOS4)

            detected_cards.append(warped_image)

    print("Mask Processing Time:", time.time() - mask_processing_start)
    print("Total Processing Time:", time.time() - start_time)

    # Return the downscaled image with edges and corners drawn
    return detected_cards


def sort_rectangle_points(points):
    """
    Sort rectangle points to ensure correct order: top-left, top-right, bottom-right, bottom-left.
    :param points: Array of 4 points (shape: (4, 2)).
    :return: Sorted points.
    """
    # Sort by y-coordinate (top to bottom)
    points = points[np.argsort(points[:, 1])]

    # Sort top points by x-coordinate (left to right)
    top_points = points[:2][np.argsort(points[:2, 0])]
    # Sort bottom points by x-coordinate (right to left)
    bottom_points = points[2:][np.argsort(points[2:, 0])][::-1]

    # Combine sorted points
    sorted_points = np.vstack((top_points, bottom_points))
    return sorted_points


def compute_robust_hash(image):
    """
    Compute multiple hashes for an image and return them as a tuple.
    :param image: PIL Image object.
    :return: Tuple of (phash, ahash, dhash).
    """
    phash = imagehash.phash(image, hash_size=32)
    ahash = imagehash.average_hash(image, hash_size=32)
    dhash = imagehash.dhash(image, hash_size=32)

    # Flatten the hashes into 1D arrays (vectors)
    phash_vec = np.array([float(i) for i in phash.hash.flatten()], dtype=np.float32)
    ahash_vec = np.array([float(i) for i in ahash.hash.flatten()], dtype=np.float32)
    dhash_vec = np.array([float(i) for i in dhash.hash.flatten()], dtype=np.float32)

    return phash_vec, ahash_vec, dhash_vec


def build_annoy_index(hash_db_path="hash_db.json", annoy_index_path="pokemon_annoy.index"):
    """
    Build and save the Annoy index from precomputed hashes.
    :param hash_db_path: Path to the JSON file containing precomputed hashes.
    :param annoy_index_path: Path to save the Annoy index.
    """
    # Load the hash database
    with open(hash_db_path, "r") as f:
        hash_db = json.load(f)

    # Create Annoy index with 768 dimensions (256 * 3 for phash, ahash, dhash)
    index = AnnoyIndex(3072, 'euclidean')  # Using 'euclidean' distance

    # Add embeddings to the Annoy index
    card_ids = []
    for i, (card_id, stored_hashes) in enumerate(hash_db.items()):
        # Convert stored hashes from strings to ImageHash objects
        stored_phash = imagehash.hex_to_hash(stored_hashes["phash"])
        stored_ahash = imagehash.hex_to_hash(stored_hashes["ahash"])
        stored_dhash = imagehash.hex_to_hash(stored_hashes["dhash"])

        # Convert hashes to embedding vectors
        phash_embedding, ahash_embedding, dhash_embedding = compute_robust_hash(stored_phash)

        # Combine all three hashes into a single embedding vector
        embedding = np.concatenate([phash_embedding, ahash_embedding, dhash_embedding])

        # Add to the Annoy index
        index.add_item(i, embedding)
        card_ids.append(card_id)

    # Build the index and save it
    index.build(10)  # Build with 10 trees
    index.save(annoy_index_path)

    return index, card_ids


def find_matching_card(detected_image, hash_db_path="hash_db.json", annoy_index_path="pokemon_annoy.index"):
    """
    Find the closest matching card ID for the detected image using Annoy index.
    :param detected_image: Detected card image as a NumPy array.
    :param hash_db_path: Path to the JSON file containing precomputed hashes.
    :param annoy_index_path: Path to the Annoy index.
    :return: ID of the closest matching card.
    """
    # Convert the detected image to PIL format and handle palette mode
    detected_pil = Image.fromarray(detected_image)
    if detected_pil.mode == 'P':
        detected_pil = detected_pil.convert('RGBA')
    detected_pil = detected_pil.convert('RGB')

    # Compute hashes for the detected image
    detected_phash, detected_ahash, detected_dhash = compute_robust_hash(detected_pil)

    # Combine all three hashes into a single embedding vector
    detected_embedding = np.concatenate([detected_phash, detected_ahash, detected_dhash])

    # Load the Annoy index
    index = AnnoyIndex(3072, 'euclidean')
    index.load(annoy_index_path)

    # Perform nearest neighbor search
    nearest_neighbors = index.get_nns_by_vector(detected_embedding, 5, include_distances=True)

    # Load the card IDs from the saved database
    with open(hash_db_path, "r") as f:
        hash_db = json.load(f)

    # Retrieve the best match from the Annoy search results
    best_match_index = nearest_neighbors[0][0]  # The best match (smallest distance)
    best_match_card_id = list(hash_db.keys())[best_match_index]

    print(f"Closest match: {best_match_card_id} with distance: {nearest_neighbors[1][0]}")
    return best_match_card_id

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React to communicate with Flask

# Your image processing code (add the code you've already written here)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Detect cards in the uploaded image
    detected_cards = detect_edges_and_corners(file.stream)

    # Find the closest match for each detected card
    matches = []
    for i, card_image in enumerate(detected_cards):
        match_id = find_matching_card(card_image)
        matches.append({'card_index': i, 'closest_match_id': match_id})

    # Return the matches as a JSON object
    return jsonify({'cards': matches})


if __name__ == '__main__':
    app.run(debug=True)
