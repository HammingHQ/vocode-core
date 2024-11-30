import time
from pathlib import Path
from typing import Dict, Optional, Union

import aiofiles
from jwcrypto import jwk, jwt
from loguru import logger


async def generate_encrypted_token(
    public_key_path: Union[str, Path] = "pubkey.pem",
    subject: str = "dev@hamming.ai", 
    expiry_hours: int = 1,
    additional_claims: Optional[Dict] = None
) -> str:
    """Generate an encrypted JWT token using RSA-OAEP and A256GCM encryption.
    
    Args:
        public_key_path: Path to the public key PEM file, as string or Path
        subject: Subject claim for the token
        expiry_hours: Number of hours until token expiration
        additional_claims: Optional additional claims to include in payload
        
    Returns:
        Encrypted JWT token string
        
    Raises:
        FileNotFoundError: If public key file not found
        jwcrypto.JWException: If token encryption fails
        ValueError: If expiry_hours is not positive
    """
    if expiry_hours <= 0:
        raise ValueError("expiry_hours must be positive")

    # Convert string path to Path object if needed
    key_path = Path(public_key_path)
    if not key_path.is_absolute():
        key_path = Path(__file__).parent / key_path

    # Validate and load the public key
    if not key_path.exists():
        raise FileNotFoundError(f"Public key not found at: {key_path}")
        
    try:
        async with aiofiles.open(key_path, mode='rb') as f:
            key_bytes = await f.read()
        public_key = jwk.JWK.from_pem(key_bytes)
    except Exception as e:
        logger.error(f"Failed to load public key: {e}")
        raise

    # Create the payload with standard claims
    current_time = int(time.time())
    claims = {
        "sub": subject,
        "exp": current_time + (expiry_hours * 3600),  # More readable than 60*60
        "iat": current_time,
        "nbf": current_time,  # Not valid before current time
    }
    
    # Add any additional claims
    if additional_claims:
        claims.update(additional_claims)

    try:
        # Create and encrypt the token
        token = jwt.JWT(
            header={"alg": "RSA-OAEP", "enc": "A256GCM"},
            claims=claims
        )
        token.make_encrypted_token(public_key)
        return token.serialize()
    except Exception as e:
        logger.error(f"Failed to create/encrypt token: {e}")
        raise
