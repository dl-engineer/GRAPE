"""Generate RSA key pair. Run once by the organizer."""
import sys
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

def main():
    out = Path(__file__).parent
    if (out / "private_key.pem").exists():
        print("private_key.pem already exists. Aborting.")
        sys.exit(1)

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    (out / "private_key.pem").write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    (out / "public_key.pem").write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    print("Keys written to encryption/private_key.pem and encryption/public_key.pem")

if __name__ == "__main__":
    main()
