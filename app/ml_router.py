import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from app.schemas import CustomerPredictionRequest, PredictionResponse

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
