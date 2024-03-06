from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import inference

app = FastAPI(openapi_url="/api/v1/ml/openapi.json", docs_url="/api/v1/ml/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(inference.router, prefix="/api-inference/v1/ml", tags=["Inference"])


@app.get("/")
async def root():
    return {"message": "ML API"}