import pandas as pd
import time
from typing import List
from fastapi import APIRouter, HTTPException, Request
from app.schemas import CustomerPredictionRequest, PredictionResponse
from src import config
from src.ml.predict import predict_batch

router = APIRouter(prefix="/ml", tags=["ML Service"])

@router.post("/predict", response_model=PredictionResponse)
async def predict_campaign_conversion(request: Request, payload: CustomerPredictionRequest):
    """
    Predicts whether a customer will subscribe to a term deposit.
    Uses the registered XGBoost model from the deployment package.
    """
    # Fetch model package from app state
    model_package = getattr(request.app.state, "model_package", None)
    
    if not model_package:
        raise HTTPException(
            status_code=503, 
            detail="ML Model package is not loaded. Train the model first."
        )
        
    try:
        pipeline = model_package["preprocessor"]
        model = model_package["model"]
        run_id = model_package.get("run_id", "local_deploy")
        
        # Convert request payload to pandas DataFrame
        # Map fields back to dots in column names as expected by feature pipeline
        data_dict = {
            "age": [payload.age],
            "job": [payload.job],
            "marital": [payload.marital],
            "education": [payload.education],
            "default": [payload.default],
            "housing": [payload.housing],
            "loan": [payload.loan],
            "contact": [payload.contact],
            "month": [payload.month],
            "day_of_week": [payload.day_of_week],
            "campaign": [payload.campaign],
            "pdays": [payload.pdays],
            "previous": [payload.previous],
            "poutcome": [payload.poutcome],
            "emp.var.rate": [payload.emp_var_rate],
            "cons.price.idx": [payload.cons_price_idx],
            "cons.conf.idx": [payload.cons_conf_idx],
            "euribor3m": [payload.euribor3m],
            "nr.employed": [payload.nr_employed]
        }
        df = pd.DataFrame(data_dict)
        
        # Preprocess features
        X_processed = pipeline.transform(df)
        
        # Predict probability
        proba_res = model.predict_proba(X_processed)
        if hasattr(proba_res, "ndim") and proba_res.ndim == 2:
            prob = float(proba_res[:, 1][0])
        else:
            prob = float(proba_res[0][1])
        pred = bool(prob >= 0.5)
        
        # Risk classification
        if prob >= 0.7:
            risk_band = "High Priority"
        elif prob >= 0.3:
            risk_band = "Medium Priority"
        else:
            risk_band = "Low Priority"
            
        # Dynamically record prediction in metric distribution
        if hasattr(request.app.state, "metrics"):
            request.app.state.metrics["prediction_distribution"][risk_band] += 1
            
        return PredictionResponse(
            subscribe_prediction=pred,
            probability=prob,
            risk_band=risk_band,
            model_version=run_id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Inference pipeline execution error: {str(e)}"
        )


@router.post("/batch-score")
async def batch_score_campaign(request: Request, payload: List[CustomerPredictionRequest]):
    """
    Score a batch of customer profiles, write the output CSV to disk, and
    return both the file path and counts categorized by priority outreach band.
    """
    # Verify model is ready
    model_package = getattr(request.app.state, "model_package", None)
    if not model_package:
        raise HTTPException(
            status_code=503,
            detail="ML Model package is not loaded. Train the model first."
        )
        
    if not payload:
        raise HTTPException(
            status_code=400,
            detail="Request body cannot be empty."
        )
        
    # Map payload objects to a list of dictionaries with standard column names
    records = []
    for p in payload:
        records.append({
            "age": p.age,
            "job": p.job,
            "marital": p.marital,
            "education": p.education,
            "default": p.default,
            "housing": p.housing,
            "loan": p.loan,
            "contact": p.contact,
            "month": p.month,
            "day_of_week": p.day_of_week,
            "campaign": p.campaign,
            "pdays": p.pdays,
            "previous": p.previous,
            "poutcome": p.poutcome,
            "emp.var.rate": p.emp_var_rate,
            "cons.price.idx": p.cons_price_idx,
            "cons.conf.idx": p.cons_conf_idx,
            "euribor3m": p.euribor3m,
            "nr.employed": p.nr_employed
        })
        
    try:
        # Run batch predictions using predict_batch module helper
        predictions = predict_batch(records)
        
        results_list = []
        counts = {
            "High Priority": 0,
            "Medium Priority": 0,
            "Low Priority": 0
        }
        
        for idx, pred in enumerate(predictions):
            prob = pred["probability"]
            if prob >= 0.7:
                risk_band = "High Priority"
            elif prob >= 0.3:
                risk_band = "Medium Priority"
            else:
                risk_band = "Low Priority"
                
            counts[risk_band] += 1
            
            # Record prediction in metric distribution
            if hasattr(request.app.state, "metrics"):
                request.app.state.metrics["prediction_distribution"][risk_band] += 1
                
            row = records[idx].copy()
            row["subscribe_prediction"] = pred["prediction"]
            row["probability"] = prob
            row["risk_band"] = risk_band
            row["model_version"] = pred["model_run_id"]
            results_list.append(row)
            
        # Write batch output to disk
        df_results = pd.DataFrame(results_list)
        timestamp = int(time.time())
        filename = f"batch_scores_{timestamp}.csv"
        out_path = config.PROCESSED_DATA_DIR / filename
        df_results.to_csv(out_path, index=False)
        
        return {
            "status": "success",
            "scored_file_path": str(out_path.resolve()),
            "total_records": len(payload),
            "counts_by_priority_band": counts
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch scoring execution failure: {str(e)}"
        )
