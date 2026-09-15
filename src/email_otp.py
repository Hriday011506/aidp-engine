import secrets
import time


def generate_otp():
    return f"{secrets.randbelow(1_000_000):06d}"


def otp_is_valid(created_at, ttl_seconds=300):
    return (time.time() - created_at) <= ttl_seconds
