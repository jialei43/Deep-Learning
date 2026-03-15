import base64
from cryptography.hazmat.primitives.asymmetric import x25519

# private_key_b64 = "eOVQU5nECRRfyfe_wa0D_EANzScaWpx2UZQ5Jbmxi2c="
# private_key_bytes = base64.urlsafe_b64decode(private_key_b64 + "=="[:len(private_key_b64)%4])
# priv_key = x25519.X25519PrivateKey.from_private_bytes(private_key_bytes)
# pub_key = priv_key.public_key()
# public_key_b64 = base64.urlsafe_b64encode(pub_key.public_bytes_raw()).decode().strip("=")
# print(f"\nYour Public Key: {public_key_b64}\n")


data = [
    {"id": 1,  "score": 90, "name": "Alice"},
    {"id": 2, "score": 95, "name": "Bob"},
    {"id": 3, "score": 95, "name": "Charlie"},
    {"id": 4, "score": 88, "name": "David"}
]

max = max(data, key=lambda x: x["score"])
print(max["name"])

