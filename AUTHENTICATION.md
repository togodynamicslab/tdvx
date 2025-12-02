# API Key Authentication & Rate Limiting

## Implementation Summary

Successfully implemented a complete API key authentication and rate limiting system for the TDvX transcription API.

## What Was Implemented

### 1. Database Layer (SQLite)
- ✅ **Location**: `app/database/`
- ✅ **Files Created**:
  - `connection.py` - SQLite connection management with context managers
  - `init_db.py` - Database schema and initialization
- ✅ **Tables**:
  - `api_keys` - Stores API key information (hashed)
  - `api_key_usage` - Tracks rate limiting per hour

### 2. Security Core
- ✅ **Location**: `app/security/`
- ✅ **Files Created**:
  - `api_keys.py` - Key generation, validation, CRUD operations
  - `rate_limiter.py` - Rate limiting logic (hourly windows)
  - `dependencies.py` - FastAPI authentication dependencies
- ✅ **Features**:
  - SHA-256 hashing for key storage
  - Cryptographically secure key generation
  - Hourly rate limiting with automatic reset
  - Master key for admin operations

### 3. Pydantic Models
- ✅ **Location**: `app/models/api_key.py`
- ✅ **Models Created**:
  - Request/response models for CRUD operations
  - Usage statistics models

### 4. Management API Endpoints
- ✅ **Location**: `app/routers/api_keys.py`
- ✅ **Endpoints** (all require master key):
  - `POST /api/keys` - Create new API key
  - `GET /api/keys` - List all keys
  - `GET /api/keys/{id}` - Get key details + usage stats
  - `PATCH /api/keys/{id}/rate-limit` - Update rate limit
  - `DELETE /api/keys/{id}` - Revoke key

### 5. Protected Endpoints
- ✅ **HTTP Endpoints** (require API key in Authorization header):
  - `GET /models`
  - `POST /transcribe`
- ✅ **WebSocket Endpoints** (require api_key query parameter):
  - `WS /transcribe/live`
  - `WS /ws/transcribe`
- ✅ **Public Endpoints** (no authentication):
  - `GET /` (index.html)
  - `GET /upload.html`
  - `GET /health`

### 6. Configuration
- ✅ Updated `app/config.py` with auth settings
- ✅ Updated `.env` with master key and configuration
- ✅ Updated `.gitignore` to exclude database files
- ✅ Fixed `requirements.txt` for torch/torchaudio compatibility

## File Structure

```
tdvx/
├── app/
│   ├── database/           [NEW]
│   │   ├── __init__.py
│   │   ├── connection.py
│   │   └── init_db.py
│   ├── security/           [NEW]
│   │   ├── __init__.py
│   │   ├── api_keys.py
│   │   ├── rate_limiter.py
│   │   └── dependencies.py
│   ├── routers/            [NEW]
│   │   ├── __init__.py
│   │   └── api_keys.py
│   ├── models/
│   │   ├── api_key.py      [NEW]
│   │   ├── response.py     [EXISTING]
│   │   └── model_config.py [EXISTING]
│   ├── config.py           [MODIFIED]
│   └── main.py             [MODIFIED]
├── data/                   [NEW]
│   └── .gitkeep
├── .env                    [MODIFIED]
├── .gitignore              [MODIFIED]
└── requirements.txt        [MODIFIED]
```

## Configuration

### Master API Key
Generated and stored in `.env`:
```
MASTER_API_KEY=master_tdvx_pqO7L2gNsx0PV4wgJ2Y4HZBNDsmou5EY-zESF7XJOs0
```

**IMPORTANT**: Keep this secure! It's used to manage API keys.

### Environment Variables

Added to `.env`:
```bash
# API Key Authentication
ENABLE_API_KEY_AUTH=true
MASTER_API_KEY=master_tdvx_pqO7L2gNsx0PV4wgJ2Y4HZBNDsmou5EY-zESF7XJOs0
DATABASE_PATH=data/tdvx.db
DEFAULT_RATE_LIMIT_PER_HOUR=100
ENABLE_USAGE_LOGGING=true
```

## Before Testing: Fix Dependency Issue

The server startup revealed a torchaudio compatibility issue (unrelated to our auth implementation).

**Run this command to fix it:**
```bash
pip uninstall -y torch torchaudio
pip install "torch>=2.0.0,<2.4.0" "torchaudio>=2.0.0,<2.4.0"
```

## How to Test

### 1. Start the Server

```bash
# Activate venv
venv\Scripts\activate

# Fix dependencies first (if needed)
pip uninstall -y torch torchaudio
pip install "torch>=2.0.0,<2.4.0" "torchaudio>=2.0.0,<2.4.0"

# Start server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Create Your First API Key

```bash
# Use the master key from .env
set MASTER_KEY=master_tdvx_pqO7L2gNsx0PV4wgJ2Y4HZBNDsmou5EY-zESF7XJOs0

# Create API key
curl -X POST "http://localhost:8000/api/keys" ^
  -H "Authorization: Bearer %MASTER_KEY%" ^
  -H "Content-Type: application/json" ^
  -d "{\"name\":\"Test Key\",\"description\":\"For testing\",\"rate_limit_per_hour\":100}"
```

**IMPORTANT**: Save the `api_key` from the response - it's shown only once!

### 3. Test Authentication

```bash
# Should fail (no auth)
curl http://localhost:8000/models

# Should succeed (with valid key)
curl http://localhost:8000/models ^
  -H "Authorization: Bearer sk_tdvx_YOUR_KEY_HERE"
```

### 4. Test Rate Limiting

```bash
# Make 101 requests (if limit=100)
for /L %i in (1,1,101) do (
  curl http://localhost:8000/models -H "Authorization: Bearer %API_KEY%"
)
```

Expected:
- First 100 requests: `200 OK`
- 101st request: `429 Too Many Requests`

### 5. Test WebSocket

```python
import asyncio
import websockets

async def test():
    api_key = "sk_tdvx_YOUR_KEY_HERE"
    uri = f"ws://localhost:8000/ws/transcribe?api_key={api_key}"

    async with websockets.connect(uri) as ws:
        print("Connected successfully!")

asyncio.run(test())
```

### 6. Test CRUD Operations

```bash
set MASTER_KEY=master_tdvx_pqO7L2gNsx0PV4wgJ2Y4HZBNDsmou5EY-zESF7XJOs0

# List keys
curl http://localhost:8000/api/keys ^
  -H "Authorization: Bearer %MASTER_KEY%"

# Get key details (replace 1 with actual ID)
curl http://localhost:8000/api/keys/1 ^
  -H "Authorization: Bearer %MASTER_KEY%"

# Update rate limit
curl -X PATCH "http://localhost:8000/api/keys/1/rate-limit" ^
  -H "Authorization: Bearer %MASTER_KEY%" ^
  -H "Content-Type: application/json" ^
  -d "{\"rate_limit_per_hour\":200}"

# Revoke key
curl -X DELETE "http://localhost:8000/api/keys/1" ^
  -H "Authorization: Bearer %MASTER_KEY%" ^
  -H "Content-Type: application/json" ^
  -d "{\"reason\":\"Testing revocation\"}"
```

## API Key Format

```
Format: sk_tdvx_<32_random_chars>
Example: sk_tdvx_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
```

- **Prefix**: `sk_tdvx_` (identifies TDvX secret key)
- **Body**: 32 cryptographically secure random characters
- **Storage**: SHA-256 hash only (never plaintext)
- **Display**: Full key shown ONCE at creation

## Security Features

1. ✅ **SHA-256 Hashing**: Keys never stored in plaintext
2. ✅ **Cryptographic Security**: Generated with `secrets.token_urlsafe()`
3. ✅ **Rate Limiting**: Hourly windows with automatic reset
4. ✅ **Master Key**: Separate authentication for admin operations
5. ✅ **Audit Trail**: Tracks creation, last use, revocation
6. ✅ **No Dependencies**: Uses only Python stdlib (sqlite3, hashlib, secrets)

## Rate Limiting Details

- **Window**: Hourly (resets at :00 of each hour)
- **Granularity**: Per API key
- **Storage**: SQLite table with UPSERT pattern
- **Response Headers**: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`
- **Error Code**: `429 Too Many Requests` when exceeded

## Database Schema

### api_keys table
```sql
CREATE TABLE api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash TEXT NOT NULL UNIQUE,          -- SHA-256
    key_prefix TEXT NOT NULL,               -- First 12 chars
    name TEXT NOT NULL,
    description TEXT,
    rate_limit_per_hour INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP,
    is_active BOOLEAN DEFAULT 1,            -- 1=active, 0=revoked
    revoked_at TIMESTAMP,
    revoked_reason TEXT
);
```

### api_key_usage table
```sql
CREATE TABLE api_key_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    api_key_id INTEGER NOT NULL,
    hour_window TEXT NOT NULL,              -- 'YYYY-MM-DD HH:00'
    request_count INTEGER DEFAULT 0,
    last_request_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (api_key_id) REFERENCES api_keys(id) ON DELETE CASCADE,
    UNIQUE(api_key_id, hour_window)
);
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'app.database'"
- Make sure all `__init__.py` files are created
- Restart the server

### "AttributeError: module 'torchaudio' has no attribute 'set_audio_backend'"
- This is a pyannote/torchaudio compatibility issue
- Fix: `pip install "torch>=2.0.0,<2.4.0" "torchaudio>=2.0.0,<2.4.0"`

### "Invalid or revoked API key"
- Check that you're using the full key (40 characters)
- Verify the key hasn't been revoked
- Check for extra spaces or newlines

### "Rate limit exceeded"
- Wait for the next hour (rate limits reset at :00)
- Or increase the rate limit via PATCH endpoint

## Next Steps

1. **Test Thoroughly**: Run all the test commands above
2. **Create Additional Keys**: For different use cases (dev, prod, etc.)
3. **Monitor Usage**: Use GET /api/keys/{id} to view statistics
4. **Production Deployment**: Use HTTPS (API keys must never be sent over HTTP)
5. **Backup Database**: Regularly backup `data/tdvx.db`

## Production Checklist

- [ ] Use HTTPS (never HTTP in production)
- [ ] Rotate master key regularly
- [ ] Set up database backups
- [ ] Monitor rate limit usage
- [ ] Set appropriate rate limits per client
- [ ] Log authentication failures
- [ ] Consider IP whitelisting for sensitive keys
- [ ] Document key management procedures for team

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Support

If you encounter issues:
1. Check the server logs for detailed error messages
2. Verify all environment variables in `.env`
3. Ensure database was initialized successfully (check startup logs)
4. Test with curl before testing with clients

## Implementation Statistics

- **Files Created**: 12
- **Files Modified**: 4
- **Lines of Code**: ~1000
- **External Dependencies Added**: 0 (uses stdlib only)
- **Time Estimated**: 5-6 hours
- **Time Actual**: Successfully completed

---

**Implementation Date**: December 2, 2025
**Status**: ✅ Complete and ready for testing
