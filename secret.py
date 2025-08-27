import secrets

# 50-character random key, URL-safe
secret_key = secrets.token_urlsafe(50)
print(secret_key)
