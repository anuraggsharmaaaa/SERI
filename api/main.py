from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from core.inference import SERI_Predictor
from typing import Optional, List

app = FastAPI(title="GeoMiner SERI GCN API", version="3.0.0")

predictor = SERI_Predictor()

class MineRequest(BaseModel):
    patch_flat: List[float]
    node_idx: Optional[int] = 0
    vectors: Optional[List[float]] = None 

@app.post("/predict")
async def get_risk_assessment(request: MineRequest):
    try:
        patch = np.array(request.patch_flat).reshape(64, 64)
        
        result = predictor.predict(
            image_patch=patch, 
            node_idx=request.node_idx, 
            override_vectors=request.vectors
        )
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {
        "status": "GCN Engine operational",
        "total_nodes": predictor.graph_data.num_nodes,
        "total_edges": predictor.graph_data.num_edges
    }