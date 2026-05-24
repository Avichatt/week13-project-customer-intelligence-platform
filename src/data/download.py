import os
import zipfile
import requests
import pandas as pd
from pathlib import Path
from src import config

def download_bank_marketing():
    print("--- Downloading UCI Bank Marketing Dataset ---")
    url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
    zip_path = config.RAW_DATA_DIR / "bank_marketing.zip"
    
    if not zip_path.exists():
        print(f"Downloading from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print("ZIP file already exists. Skipping download.")
        
    # Extract
    csv_dest = config.RAW_DATA_DIR / "bank-additional" / "bank-additional-full.csv"
    if not csv_dest.exists():
        print("Extracting ZIP file...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(config.RAW_DATA_DIR)
        print("Extraction complete.")
    else:
        print("CSV already extracted. Skipping extraction.")
    
    # Copy main file to raw root for convenience
    dest_path = config.RAW_DATA_DIR / "bank-additional-full.csv"
    src_path = config.RAW_DATA_DIR / "bank-additional" / "bank-additional-full.csv"
    if src_path.exists() and not dest_path.exists():
        import shutil
        shutil.copy(src_path, dest_path)
        print(f"Copied {src_path} to {dest_path}")
        
    print("Bank marketing dataset is ready.")

def download_cfpb_complaints(limit=1000):
    """
    Downloads consumer complaints with narratives using CFPB API.
    To avoid long runs during tests/setup, we default to 1,000 records, 
    but can request more.
    """
    print(f"--- Downloading CFPB Consumer Complaints (target: {limit}) ---")
    out_path = config.RAW_DATA_DIR / "cfpb_complaints.csv"
    
    if out_path.exists():
        print(f"{out_path} already exists. Skipping download.")
        # Load and verify count
        df = pd.read_csv(out_path)
        print(f"Existing file contains {len(df)} records.")
        return
        
    api_url = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
    
    params = {
        "has_narrative": "true",
        "size": 100,
        "field": "all",
        "sort": "created_date_desc" # Sort to get recent complaints
    }
    
    all_complaints = []
    search_after = None
    
    print("Fetching complaints...")
    while len(all_complaints) < limit:
        if search_after:
            params["search_after"] = search_after
            
        try:
            response = requests.get(api_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            print("No more hits returned from API.")
            break
            
        for hit in hits:
            src = hit.get("_source", {})
            # Make sure there is narrative content
            narrative = src.get("consumer_complaint_narrative")
            if not narrative or not isinstance(narrative, str) or len(narrative.strip()) < 10:
                continue
                
            complaint = {
                "complaint_id": src.get("complaint_id"),
                "date_received": src.get("date_received"),
                "product": src.get("product"),
                "sub_product": src.get("sub_product"),
                "issue": src.get("issue"),
                "sub_issue": src.get("sub_issue"),
                "consumer_complaint_narrative": narrative,
                "company": src.get("company"),
                "company_response": src.get("company_response_to_consumer"),
                "timely": src.get("timely_response"),
                "consumer_disputed": src.get("consumer_disputed"),
                "state": src.get("state"),
                "zip_code": src.get("zip_code")
            }
            all_complaints.append(complaint)
            if len(all_complaints) >= limit:
                break
                
        # Get sort values of the last hit for search_after
        last_sort = hits[-1].get("sort")
        if last_sort:
            import json
            search_after = json.dumps(last_sort)
        else:
            break
            
        print(f"Fetched {len(all_complaints)} / {limit} complaints...")
        
    df = pd.DataFrame(all_complaints)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} complaints with narratives to {out_path}")
    
    # Save a small sample for test/CI gate
    sample_path = config.SAMPLE_DATA_DIR / "cfpb_complaints_sample.csv"
    df.head(100).to_csv(sample_path, index=False)
    print(f"Saved 100 records sample to {sample_path}")

def make_bank_marketing_sample():
    """Create bank marketing sample for test/CI gate."""
    full_path = config.RAW_DATA_DIR / "bank-additional-full.csv"
    sample_path = config.SAMPLE_DATA_DIR / "bank-additional-full-sample.csv"
    
    if full_path.exists() and not sample_path.exists():
        df = pd.read_csv(full_path, sep=";")
        df.head(200).to_csv(sample_path, index=False, sep=";")
        print(f"Saved 200 records bank marketing sample to {sample_path}")

if __name__ == "__main__":
    download_bank_marketing()
    download_cfpb_complaints(limit=2000) # Let's fetch 2,000 for local runs, fast but sufficient
    make_bank_marketing_sample()
