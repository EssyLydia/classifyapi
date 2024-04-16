# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import JSONResponse
# from PIL import Image
# import numpy as np
# from io import BytesIO
# import pickle

# # Initialize FastAPI app
# app = FastAPI()

# # Load your trained model from pickle file
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)

# # Define function to preprocess image
# def preprocess_image(image):
#     # Resize image to the required input shape of your model
#     resized_image = image.resize((224, 224))
#     # Convert image to numpy array
#     img_array = np.array(resized_image)
#     # Normalize image data
#     img_array = img_array / 255.0
#     # Expand dimensions to match the shape expected by the model
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # Define endpoint for image classification
# @app.post("/predict/")
# async def predict_image(file: UploadFile = File(...)):
#     try:
#         # Read the image file
#         contents = await file.read()
#         image = Image.open(BytesIO(contents))
        
#         # Preprocess the image
#         processed_image = preprocess_image(image)
        
#         # Make prediction
#         prediction = model.predict(processed_image)
        
#         # Decode prediction result
#         predicted_class = int(prediction[0])
        
#         # Return the prediction as JSON response
#         return JSONResponse(content={"prediction": predicted_class})
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)})



import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
# Initialize FastAPI app
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define function to preprocess image
def preprocess_image(image):
    # Resize image to the required input shape of your model (224x224)
    resized_image = image.resize((224, 224))
    # Convert image to numpy array
    img_array = np.array(resized_image)
    # Normalize image data
    img_array = img_array / 255.0
    # Expand dimensions to match the shape expected by the model
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define endpoint for image classification
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Perform inference
        interpreter.set_tensor(input_details[0]['index'], processed_image.astype(np.float32))
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        # Decode prediction result
        predicted_class = int(np.argmax(prediction))
        
      # Map class index to class label
        class_labels = ["benign", "malignant"]
        predicted_label = class_labels[predicted_class]

        # Return the prediction as JSON response
        return JSONResponse(content={"prediction": predicted_label})
    except Exception as e:
        return JSONResponse(content={"error": str(e)})
