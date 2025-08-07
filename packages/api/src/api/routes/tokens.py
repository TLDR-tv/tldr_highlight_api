"""Token viewing endpoints."""

from datetime import datetime, timezone
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import jwt
import structlog

from shared.infrastructure.security.url_signer import JWTURLSigner
from ..dependencies import get_jwt_url_signer

logger = structlog.get_logger()

router = APIRouter()

# Set up Jinja2 templates
template_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))


@router.get("/tokens/{token}", response_class=HTMLResponse)
async def view_token_claims(
    request: Request,
    token: str,
    jwt_signer: JWTURLSigner = Depends(get_jwt_url_signer),
):
    """View JWT token claims in a formatted HTML page.
    
    This endpoint accepts a JWT token and displays its decoded claims
    in a user-friendly HTML format. Useful for debugging and verification.
    """
    try:
        # Decode the token without verification first to get all claims
        unverified_claims = jwt.decode(token, options={"verify_signature": False})
        
        # Check if token is expired
        now = datetime.now(timezone.utc)
        exp_timestamp = unverified_claims.get("exp")
        is_expired = False
        
        if exp_timestamp:
            exp_datetime = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
            is_expired = now > exp_datetime
        
        # Format timestamps for display
        iat_timestamp = unverified_claims.get("iat")
        issued_at_formatted = None
        expires_at_formatted = None
        
        if iat_timestamp:
            iat_datetime = datetime.fromtimestamp(iat_timestamp, tz=timezone.utc)
            issued_at_formatted = iat_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")
            
        if exp_timestamp:
            exp_datetime = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
            expires_at_formatted = exp_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Extract additional claims (everything not in standard set)
        standard_claims = {
            "sub", "resource_type", "iat", "exp", "type", 
            "stream_id", "stream_fingerprint", "highlight_ids"
        }
        additional_claims = {
            k: v for k, v in unverified_claims.items() 
            if k not in standard_claims
        }
        
        # Prepare template context
        context = {
            "request": request,
            "claims": unverified_claims,
            "is_expired": is_expired,
            "issued_at_formatted": issued_at_formatted,
            "expires_at_formatted": expires_at_formatted,
            "additional_claims": additional_claims if additional_claims else None,
            "raw_token": token,
        }
        
        logger.info(
            "Token claims viewed",
            organization_id=unverified_claims.get("sub"),
            resource_type=unverified_claims.get("resource_type"),
            is_expired=is_expired
        )
        
        return templates.TemplateResponse("token_claims.html", context)
        
    except jwt.DecodeError:
        logger.warning("Invalid JWT token format provided")
        raise HTTPException(
            status_code=400,
            detail="Invalid JWT token format"
        )
    except Exception as e:
        logger.error("Error processing token", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Error processing token"
        )