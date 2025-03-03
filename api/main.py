# main.py

import xgboost as xgb
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import uvicorn


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = None


@app.on_event("startup")
def load_model():
    global model
    model = xgb.XGBRegressor()
    model.load_model("models/xgboost_scheduler_v1.json")


@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()

        # Extract features safely
        features = data.get("features")
        print("Received data:", features)

        if features is None or not isinstance(features, list):
            return {
                "error": "Invalid input format. 'features' must be a list of numbers."
            }

        # Ensure all elements are numbers
        try:
            features = [float(x) for x in features]
        except ValueError:
            return {"error": "All elements in 'features' must be numeric values."}

        # Convert to a NumPy array and reshape
        np_features = np.array(features, dtype=np.float32).reshape(1, -1)

        print("Processed np_features shape:", np_features.shape)

        # Predict
        prediction = model.predict(np_features)

        return {"prediction": prediction.tolist()}

    except Exception as e:
        print("Error during prediction:", str(e))
        return {"error": str(e)}


@app.get("/")
async def healthcheck():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)
