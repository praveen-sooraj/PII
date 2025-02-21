from fastapi import FastAPI, Request
from utils import classify_sector_with_alpaca, detect_entities_from_dynamic_json, fetch_standards_and_check_violations, analyze_dynamic_ids
import json

app=FastAPI()

@app.post("/process_pii")
async def get_sector(data: Request):
    try:
        operation_id = data.get("id")
        workspace_id = data.get("workspace_id")
        collection_id = data.get("collection_id")
        project_id = data.get("project_id")
        tenant_id = data.get("tenant_id")

        sector = classify_sector_with_alpaca(str(data))
        entities = detect_entities_from_dynamic_json(json.dumps(data), sector)
        new_detection = fetch_standards_and_check_violations(entities, tenant_id)
        id_check_results = analyze_dynamic_ids(data)

        processed_result = {
            "sector": sector,
            "pii_entities": entities,
            "standard_check": new_detection,
            "id_check_results": id_check_results
        }

        print(f"Processed Result: {json.dumps(processed_result, indent=2)}")

    except Exception as e:
        print(f"Error during processing: {str(e)}")