# app/utils/response_helper_middleware.py
from fastapi.responses import JSONResponse
from time import time
from app.models.global_response import GlobalResponse, ErrorItem, Meta

def success_response(data=None, message="Success", code="OK", status_code=200, meta=None, process_time=None):
    payload = GlobalResponse(
        status="success",
        status_code=status_code,
        message=message,
        data=data,
        errors=None,
        process_time=process_time
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())

def error_response(message="Error", code="ERROR", status_code=400, errors=None, process_time=None):
    error_items = [ErrorItem(**e) if isinstance(e, dict) else e for e in (errors or [])]
    payload = GlobalResponse(
        status="error",
        status_code=status_code,
        
        message=message,
        data=None,
        errors=error_items or None,
    
        process_time=process_time
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())



# New helper: returns a plain dict for use in HTTPException detail
def error_response_dict(message="Error", code="ERROR", status_code=400, errors=None, process_time=None):
    error_items = [ErrorItem(**e) if isinstance(e, dict) else e for e in (errors or [])]
    payload = GlobalResponse(
        status="error",
        status_code=status_code,
        message=message,
        data=None,
        errors=error_items or None,
        process_time=process_time
    )
    return payload.model_dump()  # Return dict, no JSONResponse wrapping