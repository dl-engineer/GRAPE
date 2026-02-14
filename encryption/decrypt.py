"""Decrypt a .enc submission using the private RSA key (hybrid RSA+Fernet)."""
import os, sys
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.fernet import Fernet

RSA_BLOCK = 256  # 2048-bit key


def decrypt_file(enc_path: str, private_key_pem: str | None = None) -> bytes:
    if private_key_pem is None:
        private_key_pem = os.environ.get("SUBMISSION_PRIVATE_KEY", "")
    private_key_pem = private_key_pem.replace("\\n", "\n").strip()
    if not private_key_pem:
        raise ValueError("No private key provided.")

    priv = serialization.load_pem_private_key(private_key_pem.encode(), password=None)
    raw = open(enc_path, "rb").read()

    session_key = priv.decrypt(
        raw[:RSA_BLOCK],
        padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
    )
    return Fernet(session_key).decrypt(raw[RSA_BLOCK:])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python encryption/decrypt.py <file.enc> [output.csv]")
        sys.exit(1)
    data = decrypt_file(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace(".enc", "")
    open(out, "wb").write(data)
    print(f"Decrypted -> {out}")
