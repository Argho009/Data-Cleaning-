# Proxy to root app.py to satisfy OpenEnv validate
from app import app, main as original_main

def main() -> None:
    original_main()

if __name__ == "__main__":
    main()
