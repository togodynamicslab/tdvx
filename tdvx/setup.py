"""
Setup script to verify environment and download models
"""
import os
import sys


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_env_file():
    """Check if .env file exists"""
    if not os.path.exists('.env'):
        print("⚠️  .env file not found")
        print("   Creating from .env.example...")
        if os.path.exists('.env.example'):
            with open('.env.example', 'r') as src:
                content = src.read()
            with open('.env', 'w') as dst:
                dst.write(content)
            print("✅ .env file created")
            print("   ⚠️  Please edit .env and add your PYANNOTE_AUTH_TOKEN")
            return False
        else:
            print("❌ .env.example not found")
            return False
    print("✅ .env file exists")
    return True


def check_token():
    """Check if Pyannote token is configured"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        token = os.getenv('PYANNOTE_AUTH_TOKEN')
        if not token or token == 'your_huggingface_token_here':
            print("⚠️  PYANNOTE_AUTH_TOKEN not configured in .env")
            print("   Get your token from: https://huggingface.co/settings/tokens")
            print("   Accept model license: https://huggingface.co/pyannote/speaker-diarization-3.1")
            return False
        print("✅ Pyannote token configured")
        return True
    except ImportError:
        print("⚠️  python-dotenv not installed yet")
        return True  # Will be installed with requirements


def check_ffmpeg():
    """Check if FFmpeg is installed"""
    import subprocess
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ FFmpeg installed")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("⚠️  FFmpeg not found")
    print("   Install with: choco install ffmpeg (Windows)")
    print("   Or visit: https://ffmpeg.org/download.html")
    return False


def check_torch():
    """Check PyTorch and CUDA availability"""
    try:
        import torch
        print(f"✅ PyTorch installed: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available (CPU mode)")
            print("   For GPU support, install CUDA-enabled PyTorch:")
            print("   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return True
    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return True  # Will be installed with requirements


def main():
    print("=" * 60)
    print("Live Transcription API - Setup Verification")
    print("=" * 60)
    print()

    checks = [
        ("Python Version", check_python_version),
        ("Environment File", check_env_file),
        ("FFmpeg", check_ffmpeg),
        ("PyTorch", check_torch),
        ("Pyannote Token", check_token),
    ]

    results = []
    for name, check_func in checks:
        print(f"Checking {name}...")
        results.append(check_func())
        print()

    print("=" * 60)
    if all(results):
        print("✅ All checks passed! You're ready to go!")
        print()
        print("Next steps:")
        print("1. Run: python app/main.py")
        print("2. Open: test_client.html in your browser")
    else:
        print("⚠️  Some checks failed. Please review the messages above.")
        print()
        print("Common next steps:")
        print("1. pip install -r requirements.txt")
        print("2. Edit .env and add your PYANNOTE_AUTH_TOKEN")
        print("3. Install FFmpeg if needed")

    print("=" * 60)


if __name__ == "__main__":
    main()
