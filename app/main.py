from fastapi import FastAPI

app = FastAPI(title="Dataset Doctor")


@app.get("/")
def home():
    return {"status": "Dataset Doctor is running"}
