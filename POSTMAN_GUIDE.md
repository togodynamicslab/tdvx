# Postman Collection Guide

## Quick Start

### 1. Import the Collection

1. Open Postman
2. Click **Import** button (top left)
3. Select **File** tab
4. Choose `TDvX_API.postman_collection.json`
5. Click **Import**

### 2. Collection Structure

The collection is organized into folders:

```
TDvX Transcription API
├── 📁 Authentication & Keys (Master key required)
│   ├── Create API Key
│   ├── List API Keys
│   ├── Get API Key Details
│   ├── Update Rate Limit
│   └── Revoke API Key
│
├── 📁 Transcription (API key required)
│   ├── List Available Models
│   ├── Transcribe Audio File
│   ├── Transcribe File (High Quality)
│   └── Transcribe File (Balanced)
│
├── 📁 Public Endpoints (No auth)
│   ├── Health Check
│   └── API Documentation
│
└── 📁 Testing & Examples
    ├── Test - No Auth (Should Fail)
    ├── Test - Invalid Key (Should Fail)
    └── Test - Rate Limit
```

### 3. Setup Environment Variables

The collection comes with pre-configured variables:

| Variable | Default Value | Description |
|----------|--------------|-------------|
| `base_url` | `http://localhost:8000` | API server URL |
| `master_key` | `master_tdvx_pqO...` | Master key for managing API keys |
| `api_key` | *(empty)* | Your API key (auto-filled after creation) |
| `key_id` | *(empty)* | Your API key ID (auto-filled) |

**Optional**: Create a Postman Environment for better organization:
1. Click **Environments** (left sidebar)
2. Click **Create Environment**
3. Name it "TDvX Local"
4. Add the variables above
5. Select it in the top-right dropdown

### 4. Basic Workflow

#### Step 1: Create Your First API Key

1. Open **Authentication & Keys** → **Create API Key**
2. Click **Send**
3. ✅ The `api_key` will be automatically saved to your variables!
4. ⚠️ **IMPORTANT**: Copy the full API key from the response - it's shown only once!

Example response:
```json
{
  "id": 1,
  "api_key": "sk_tdvx_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
  "key_prefix": "sk_tdvx_a1b2",
  "name": "Test API Key",
  "rate_limit_per_hour": 100,
  "created_at": "2025-12-02T16:30:00"
}
```

#### Step 2: Test Authentication

1. Open **Transcription** → **List Available Models**
2. Click **Send**
3. ✅ Should return list of models

#### Step 3: Transcribe Audio

1. Open **Transcription** → **Transcribe Audio File**
2. Go to **Body** → **form-data**
3. Click on **Select Files** for the `file` field
4. Choose an audio file (WAV, MP3, M4A, FLAC)
5. Click **Send**
6. ✅ Should return transcription with speaker labels

## Features

### 🔄 Auto-Save API Key

When you create a new API key, the collection automatically:
- Saves `api_key` to variables
- Saves `key_id` to variables
- Logs the key to console

Check the **Console** (bottom panel) to see the saved values.

### 🧪 Built-in Tests

Each request includes tests that:
- ✅ Validate response status codes
- ✅ Check for required fields
- ✅ Log important information to console

View test results in the **Test Results** tab after sending a request.

### 📊 Rate Limit Tracking

The "Test - Rate Limit" request shows remaining requests:
- Run it multiple times
- Check console for remaining count
- When you hit the limit, you'll get `429 Too Many Requests`

### 🔐 Pre-configured Authentication

All requests have the correct authentication:
- **Master key requests**: Use `{{master_key}}` variable
- **API key requests**: Use `{{api_key}}` variable
- **Public endpoints**: No authentication needed

## Example Requests

### Create a Production API Key

```http
POST {{base_url}}/api/keys
Authorization: Bearer {{master_key}}
Content-Type: application/json

{
  "name": "Production Key",
  "description": "For production use",
  "rate_limit_per_hour": 1000
}
```

### List All Keys (Including Revoked)

```http
GET {{base_url}}/api/keys?include_revoked=true
Authorization: Bearer {{master_key}}
```

### Update Rate Limit

```http
PATCH {{base_url}}/api/keys/{{key_id}}/rate-limit
Authorization: Bearer {{master_key}}
Content-Type: application/json

{
  "rate_limit_per_hour": 200
}
```

### Transcribe with Specific Model

```http
POST {{base_url}}/transcribe?model=tdv1
Authorization: Bearer {{api_key}}
Content-Type: multipart/form-data

file: [your_audio_file.wav]
```

## Testing Scenarios

### Test 1: No Authentication

1. Run **Testing & Examples** → **Test - No Auth (Should Fail)**
2. ✅ Should return `401 Unauthorized`
3. ✅ Test will pass if authentication is working

### Test 2: Invalid Key

1. Run **Testing & Examples** → **Test - Invalid Key (Should Fail)**
2. ✅ Should return `401 Unauthorized`
3. ✅ Test will pass if validation is working

### Test 3: Rate Limiting

1. Run **Testing & Examples** → **Test - Rate Limit** 100+ times
2. First 100: ✅ `200 OK`
3. After that: ✅ `429 Too Many Requests`
4. Check response headers:
   - `X-RateLimit-Limit`: Your limit
   - `X-RateLimit-Remaining`: Requests left
   - `X-RateLimit-Reset`: When it resets

### Test 4: Full Workflow

1. **Create Key**: Run "Create API Key"
2. **List Keys**: Run "List API Keys" → Should see your key
3. **Get Details**: Run "Get API Key Details" → Should see usage stats
4. **Update Limit**: Run "Update Rate Limit" → Change to 200/hour
5. **Use Key**: Run "List Available Models" → Should work
6. **Revoke**: Run "Revoke API Key" → Key becomes invalid
7. **Test Revoked**: Run "List Available Models" → Should fail with 401

## Tips & Tricks

### 💡 Use Collection Runner

Run multiple tests automatically:
1. Click **...** on collection
2. Select **Run collection**
3. Choose requests to run
4. Click **Run TDvX Transcription API**
5. View results summary

### 💡 Copy as cURL

Convert any request to cURL:
1. Click **Code** (right side, under Send)
2. Select **cURL**
3. Copy and use in terminal

### 💡 Monitor Rate Limits

Keep an eye on headers:
1. Open **Response** → **Headers**
2. Look for `X-RateLimit-*` headers
3. Track your usage

### 💡 Save Responses as Examples

Save successful responses:
1. Send request
2. Click **Save Response**
3. Choose **Save as Example**
4. Helps document API behavior

## Troubleshooting

### ❌ "Could not send request"

**Problem**: Server not running

**Solution**:
```bash
cd C:\Users\Usuario\war\tdvx
venv\Scripts\activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### ❌ "401 Unauthorized" on authenticated requests

**Problem**: API key not set or expired

**Solution**:
1. Check `{{api_key}}` variable is filled
2. Create new key if needed
3. Verify key wasn't revoked

### ❌ "429 Too Many Requests"

**Problem**: Rate limit exceeded

**Solution**:
1. Wait for next hour (resets at :00)
2. Or update rate limit via "Update Rate Limit"
3. Or create new key with higher limit

### ❌ Variables not updating

**Problem**: Using Collection variables instead of Environment

**Solution**:
1. Check top-right dropdown
2. Make sure correct environment is selected
3. Or use Collection variables (they work too!)

### ❌ File upload fails

**Problem**: File too large or wrong format

**Solution**:
1. Check file size < 100MB (default limit)
2. Supported formats: WAV, MP3, M4A, FLAC
3. Check server logs for detailed error

## Advanced Usage

### Using in CI/CD

Export collection and run with Newman:
```bash
# Install Newman
npm install -g newman

# Run collection
newman run TDvX_API.postman_collection.json \
  --env-var "base_url=http://localhost:8000" \
  --env-var "master_key=master_tdvx_xxx" \
  --reporters cli,json
```

### Monitor API in Production

1. **Postman Monitor**: Schedule collection runs
2. **Webhooks**: Trigger on events
3. **Integrations**: Connect to Slack, PagerDuty, etc.

### Generate Documentation

1. Click **...** on collection
2. Select **View documentation**
3. Click **Publish**
4. Share public documentation

## Support

If you encounter issues:

1. **Check Console**: View detailed logs (bottom panel)
2. **Check Tests**: See which assertions failed
3. **Check Server Logs**: Look at server output
4. **Check Environment**: Verify variables are set correctly

## Useful Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + Enter` | Send request |
| `Ctrl + S` | Save request |
| `Ctrl + Alt + C` | Copy request as cURL |
| `Ctrl + /` | Toggle sidebar |
| `Alt + Shift + F` | Format JSON body |

---

**Happy Testing!** 🚀

For more information, see `AUTHENTICATION.md`
