from google import genai
import sys

client = genai.Client(api_key="AIzaSyAY3YKskvTmzkYZL-Eu5IXbEUc8z25myzw")

try:
    print("Testing Key...")
    sys.stdout.flush()
    response = client.models.generate_content(
        model="gemini-2.5-flash",  # Updated to use available model
        contents="Hello, are you working?"
    )
    print("Success! Key is valid.")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Key Failed: {e}")
    import traceback
    traceback.print_exc()