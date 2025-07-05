import os
import requests
from dotenv import load_dotenv

def test_llm_connection():
    """Test LLM API connections"""
    # Load environment variables
    load_dotenv()
    
    # Test Together.ai API
    together_key = os.getenv("TOGETHER_API_KEY")
    if together_key:
        print("\n🔄 Testing Together.ai API connection...")
        result = test_together_ai(together_key)
        if result:
            print("✅ Together.ai API connection successful!")
            return True
    else:
        print("❌ Together.ai API key not found in environment variables")
    
    # Test Anthropic API
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        print("\n🔄 Testing Anthropic API connection...")
        result = test_anthropic(anthropic_key)
        if result:
            print("✅ Anthropic API connection successful!")
            return True
    else:
        print("❌ Anthropic API key not found in environment variables")
    
    print("\n❌ No working API connections found. The app will run in offline mode.")
    return False

def test_together_ai(api_key):
    """Test Together.ai API"""
    endpoint = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [
            {"role": "system", "content": "You are a helpful dermatology expert."},
            {"role": "user", "content": "What is hormonal acne?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    try:
        response = requests.post(endpoint, headers=headers, json=data, timeout=15)
        if response.status_code == 200:
            response_data = response.json()
            message = response_data['choices'][0]['message']['content']
            print(f"Sample response: {message[:100]}...")
            return True
        else:
            print(f"❌ Error: API request failed with status code {response.status_code}")
            if response.text:
                try:
                    error_data = response.json()
                    print(f"Error details: {error_data}")
                except:
                    print(f"Error response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_anthropic(api_key):
    """Test Anthropic API"""
    endpoint = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    data = {
        "model": "claude-3-sonnet-20240229",
        "messages": [
            {
                "role": "user",
                "content": "What is hormonal acne?"
            }
        ],
        "max_tokens": 100
    }
    
    try:
        response = requests.post(endpoint, headers=headers, json=data, timeout=15)
        if response.status_code == 200:
            response_data = response.json()
            message = response_data['content'][0]['text']
            print(f"Sample response: {message[:100]}...")
            return True
        else:
            print(f"❌ Error: API request failed with status code {response.status_code}")
            if response.text:
                try:
                    error_data = response.json()
                    print(f"Error details: {error_data}")
                except:
                    print(f"Error response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_llm_connection() 