import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load preprocessed data
images = np.load("train_images.npy")
labels = np.load("train_labels.npy")

X, y = np.array(images), np.array(labels)

# Normalize and reshape
X = X.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y = tf.keras.utils.to_categorical(y, 10)
# Split: 75% train, 15% val, 10% test
total = len(X)
train_end = int(total * 0.75)
val_end = int(total * 0.90)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# Build the CNN model
model = models.Sequential([
    layers.InputLayer(input_shape=(28, 28, 1)),

    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # Flatten and dense layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 classes for digits
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_val, y_val, verbose=2)
print(f"Validation accuracy: {test_acc * 100:.2f}%")


# Function to improve recognition of the digit 4
def preprocess_digit_4(image):
    """Apply special preprocessing for digit 4"""
    # Make a copy to avoid modifying the original
    processed = image.copy()

    # Apply additional processing that might help with digit 4
    # 1. Enhance vertical lines (common in 4s)
    kernel_vertical = np.ones((3, 1), np.uint8)
    processed = cv2.dilate(processed, kernel_vertical, iterations=1)

    # 2. Close small
    kernel_close = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_close)

    # 3. Thin horizontal lines (to emphasize vertical structure of 4)
    kernel_horizontal = np.ones((1, 2), np.uint8)
    processed = cv2.erode(processed, kernel_horizontal, iterations=1)

    return processed


# Prediction on drawn digits
def predict_drawn_digit(model):
    canvas = np.zeros((280, 280), dtype='uint8')  # Black canvas
    drawing = False
    last_point = None
    result_str = ""  # Initialize empty result string

    def draw(event, x, y, flags, param):
        nonlocal drawing, last_point
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.line(canvas, last_point, (x, y), 255, 15)  # Increased line thickness
            last_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.namedWindow("Draw Digit")
    cv2.setMouseCallback("Draw Digit", draw)

    processed_window_created = False
    prediction_history = []  # Store all predictions to build final string

    while True:
        cv2.imshow("Draw Digit", canvas)
        key = cv2.waitKey(1)

        if key == ord('c'):  # Clear
            canvas[:] = 0
            print("Canvas cleared")

            # Close processed digit window if open
            if processed_window_created:
                cv2.destroyWindow("Processed Digit")
                processed_window_created = False

        elif key == ord('p'):  # Predict
            # Find bounding box of the digit
            digit_img = canvas.copy()

            # Find non-zero pixels (the drawing)
            if np.sum(digit_img) > 0:  # Check if drawing exists
                rows, cols = np.nonzero(digit_img)

                # Get bounding box
                top, bottom = np.min(rows), np.max(rows)
                left, right = np.min(cols), np.max(cols)

                # Add small padding
                padding = 20
                top = max(0, top - padding)
                bottom = min(digit_img.shape[0] - 1, bottom + padding)
                left = max(0, left - padding)
                right = min(digit_img.shape[1] - 1, right + padding)

                # Extract the digit
                extracted_digit = digit_img[top:bottom + 1, left:right + 1]

                # Create a square image with padding
                square_size = max(extracted_digit.shape[0], extracted_digit.shape[1])
                square_img = np.zeros((square_size, square_size), dtype='uint8')

                # Center the digit in the square
                y_offset = (square_size - extracted_digit.shape[0]) // 2
                x_offset = (square_size - extracted_digit.shape[1]) // 2
                square_img[y_offset:y_offset + extracted_digit.shape[0],
                x_offset:x_offset + extracted_digit.shape[1]] = extracted_digit

                # Resize to 20x20 (MNIST standard is to have 20x20 images with 4px padding)
                img = cv2.resize(square_img, (20, 20))

                # Add 4px padding to make it 28x28
                img_padded = np.zeros((28, 28), dtype='uint8')
                img_padded[4:24, 4:24] = img

                # Apply Gaussian blur to reduce noise
                img_padded = cv2.GaussianBlur(img_padded, (3, 3), 0)

                # Show processed image in a separate window
                cv2.imshow("Processed Digit", cv2.resize(img_padded, (140, 140)))
                processed_window_created = True

                # Check if the prediction might be a 4 and apply special processing
                # Try prediction with normal processing first
                img_norm = img_padded.astype('float32') / 255.0
                img_norm = img_norm.reshape(1, 28, 28, 1)

                # Get prediction probabilities
                pred_probs = model.predict(img_norm, verbose=0)[0]

                # If one of the top predictions is 4 but not confident, try special processing
                top3_indices = np.argsort(pred_probs)[-3:][::-1]
                if 4 in top3_indices and pred_probs[4] < 0.8:
                    # Apply special processing for digit 4
                    img_special = preprocess_digit_4(img_padded)
                    img_special_norm = img_special.astype('float32') / 255.0
                    img_special_norm = img_special_norm.reshape(1, 28, 28, 1)

                    # Get new prediction with special processing
                    special_pred_probs = model.predict(img_special_norm, verbose=0)[0]

                    # If confidence for digit 4 improved, use this prediction instead
                    if special_pred_probs[4] > pred_probs[4]:
                        pred_probs = special_pred_probs
                        print("Used special processing for digit 4")

                # Get top 3 predictions
                top3_indices = np.argsort(pred_probs)[-3:][::-1]  # Sort and get top 3
                top3_probs = pred_probs[top3_indices]

                # Get the top prediction
                pred = top3_indices[0]

                # Add this prediction to history
                prediction_history.append(str(pred))

                # Update the result string by joining all predictions
                result_str = ''.join(prediction_history)

                # Display top 3 predictions with confidences
                print("Top 3 predictions:")
                for i, (idx, prob) in enumerate(zip(top3_indices, top3_probs)):
                    print(f"  #{i + 1}: Digit {idx} with {prob * 100:.2f}% confidence")

                print(f"Current sequence: {result_str}")

                # Clear canvas for next digit
                canvas[:] = 0

            else:
                print("Nothing to predict. Please draw a digit first.")

        elif key == ord('r'):  # Reset the result string
            result_str = ""
            prediction_history = []
            print("Result string reset")

        elif key == ord('q'):  # Quit
            break

    cv2.destroyAllWindows()
    print("Final predicted digits:", result_str)
    return result_str


# Now run the prediction function
predict_drawn_digit(model)