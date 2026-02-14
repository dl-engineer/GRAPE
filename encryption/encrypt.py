"""Encrypt a submission CSV using the public RSA key (hybrid RSA+Fernet)."""
import sys
from pathlib import Path
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.fernet import Fernet


def encrypt_file(csv_path: str, out_path: str | None = None):
    key_path = Path(__file__).parent / "public_key.pem"
    pub_key = serialization.load_pem_public_key(key_path.read_bytes())

    session_key = Fernet.generate_key()
    encrypted_data = Fernet(session_key).encrypt(Path(csv_path).read_bytes())

    encrypted_session_key = pub_key.encrypt(
        session_key,
        padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
    )

    dest = Path(out_path) if out_path else Path(csv_path).with_suffix(".csv.enc")
    dest.write_bytes(encrypted_session_key + encrypted_data)
    print(f"Encrypted -> {dest}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python encryption/encrypt.py <predictions.csv> [output.enc]")
        sys.exit(1)
    encrypt_file(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
