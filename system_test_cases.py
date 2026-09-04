from app import app
sample_aud = "dataset/16000_pcm_speeches/test_audio/sample.wav"
test_cases = [
    {
        "id": "ST01",
        "scenario": "Open Home Page",
        "input": "GET /",
        "expected": "Home page loads successfully"
    },
    {
        "id": "ST02",
        "scenario": "Open Recognition Page",
        "input": "GET /rec.html",
        "expected": "Recognition page loads"
    },
    {
        "id": "ST03",
        "scenario": "Upload audio file for recognition",
        "input": "sample_aud",
        "expected": "Speaker name displayed"
    }
]

client = app.test_client()

print("\nSYSTEM TEST RESULTS\n")
print("{:<10} {:<35} {:<25} {:<30} {:<10}".format(
    "Test ID","Scenario","Input","Expected Output","Result"))
print("-"*120)

for test in test_cases:

    try:
        if test["id"] == "ST01":
            response = client.get("/")
            result = "Pass" if response.status_code == 200 else "Fail"

        elif test["id"] == "ST02":
            response = client.get("/rec.html")
            result = "Pass" if response.status_code == 200 else "Fail"

        elif test["id"] == "ST03":
            with open(sample_aud, "rb") as audio:
                data = {"file": (audio, "sample_aud")}
                response = client.post("/rec.html", data=data)

            result = "Pass" if response.status_code == 200 else "Fail"

        print("{:<10} {:<35} {:<25} {:<30} {:<10}".format(
            test["id"],
            test["scenario"],
            test["input"],
            test["expected"],
            result
        ))

    except:
        print("{:<10} {:<35} {:<25} {:<30} {:<10}".format(
            test["id"],
            test["scenario"],
            test["input"],
            test["expected"],
            "Fail"
        ))