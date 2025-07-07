from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Bias Checker is running"}
