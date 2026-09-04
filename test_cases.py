import numpy as np
from app import features_extractor

test_cases = [
    {
        "id": "UT01",
        "scenario": "Test audio file upload and feature extraction",
        "input": "dataset/16000_pcm_speeches/test_audio/sample.wav",
        "expected": "MFCC features extracted"
    },
    {
        "id": "UT02",
        "scenario": "Check MFCC feature length",
        "input": "dataset/16000_pcm_speeches/test_audio/sample1.wav",
        "expected": "Feature length = 40"
    }
]

print("\nUNIT TEST RESULTS\n")
print("{:<10} {:<40} {:<15} {:<25} {:<10}".format(
    "Test ID", "Scenario", "Input", "Expected Output", "Result"))

print("-"*110)

for test in test_cases:
    try:
        features = features_extractor(test["input"])

        if test["id"] == "UT01":
            result = "Pass" if isinstance(features, np.ndarray) else "Fail"

        elif test["id"] == "UT02":
            result = "Pass" if len(features) == 40 else "Fail"

        print("{:<10} {:<40} {:<15} {:<25} {:<10}".format(
            test["id"],
            test["scenario"],
            test["input"],
            test["expected"],
            result
        ))

    except Exception as e:
        print("{:<10} {:<40} {:<15} {:<25} {:<10}".format(
            test["id"],
            test["scenario"],
            test["input"],
            test["expected"],
            "Fail"
        ))