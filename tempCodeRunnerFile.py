import requests

fake_article = """
Scientists have NEVER been able to explain why the moon is hollow.
According to anonymous sources, NASA has been hiding alien civilizations
living inside the moon since 1969. The government doesn't want you to know this.
"""

real_article = """
The Federal Reserve raised interest rates by 25 basis points on Wednesday,
the tenth increase in the current tightening cycle. Fed Chair Jerome Powell
said in a press conference that inflation has moderated but remains above the
2% target, and that further policy decisions will depend on incoming data.
"""

for label, article in [("should be FAKE", fake_article), ("should be REAL", real_article)]:
    r = requests.post(
        "http://127.0.0.1:5000/predict",
        json={"article": article}
    )
    data = r.json()
    print(f"\nTest [{label}]")
    print(f"  Result     : {data['prediction']}")
    print(f"  Confidence : {data['confidence']}%")
    print(f"  Fake prob  : {data['fake_prob']}%")
    print(f"  Real prob  : {data['real_prob']}%")