import subprocess
import json
import time

def update_threshold(value):
    """Update .env file with new threshold"""
    with open('.env', 'r') as f:
        lines = f.readlines()

    with open('.env', 'w') as f:
        for line in lines:
            if line.startswith('PYANNOTE_CLUSTERING_THRESHOLD='):
                f.write(f'PYANNOTE_CLUSTERING_THRESHOLD={value}\n')
            else:
                f.write(line)

def test_threshold(threshold):
    """Test a specific threshold value"""
    print(f'\nTesting threshold: {threshold}')
    print('-' * 50)

    # Update .env
    update_threshold(threshold)
    print(f'Updated .env with threshold={threshold}')

    # Start server
    print('Starting server...')
    server = subprocess.Popen(
        'venv\\Scripts\\activate && uvicorn app.main:app --host 0.0.0.0 --port 8000',
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Wait for server to start
    time.sleep(20)

    # Test audio
    output_file = f'test_threshold_{threshold}.json'
    print(f'Testing audio3.mp3...')
    subprocess.run(
        f'curl -s -X POST "http://localhost:8000/transcribe?model=tdv1-fast" -F "file=@C:/Users/Matheus/Downloads/audio3.mp3" -o {output_file}',
        shell=True,
        timeout=90
    )

    # Count speakers
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        speakers = set(seg['speaker'] for seg in data['segments'])
        num_speakers = len(speakers)
        print(f'Result: {num_speakers} speakers - {sorted(speakers)}')
        result = {'threshold': threshold, 'speakers': num_speakers, 'speaker_list': sorted(speakers)}
    except Exception as e:
        print(f'Error: {e}')
        result = {'threshold': threshold, 'error': str(e)}

    # Kill server
    server.terminate()
    try:
        server.wait(timeout=5)
    except:
        server.kill()

    time.sleep(3)

    return result

# Test different thresholds
thresholds = [0.6, 0.7, 0.8, 0.9]
results = []

print('=' * 50)
print('TESTING CLUSTERING THRESHOLDS')
print('=' * 50)

for t in thresholds:
    result = test_threshold(t)
    results.append(result)

# Print summary
print('\n' + '=' * 50)
print('SUMMARY')
print('=' * 50)
for r in results:
    if 'error' in r:
        print(f'Threshold {r["threshold"]}: ERROR')
    else:
        print(f'Threshold {r["threshold"]}: {r["speakers"]} speakers')

print(f'\nTarget: 8 speakers')
