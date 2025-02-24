import os
import openai

# 1. Configure Your OpenAI API Key. Use environment variable or assign directly.
openai.api_key = os.getenv("OPENAI_API_KEY")

# 2. Define the sentiment-analysis function
def analyze_sentiment(message):
    chat_messages = [
        {
            "role": "system",
            "content": "You analyze the emotional tone of messages."
        },
        {
            "role": "user",
            "content": (
                f"Analyze the following message and summarize its "
                f"emotional tone:\n\n", 
                f"\"{message}\"\n\n"
                f"Categorize it as positive, neutral, or negative, "
                f"and provide any nuanced feelings you detect."
            )
        }
    ]
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=chat_messages,
        max_tokens=4096,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# 3. Define the empathy-response function
def generate_empathy_response(message, sentiment_summary):
    chat_messages = [
        {
            "role": "system",
            "content": "You craft empathetic and helpful responses to messages."
        },
        {
            "role": "user",
            "content": (
                f"Craft a kind and helpful response to the following message:\n\n"
                f"\"{message}\"\n\n"
                f"The emotional tone is: {sentiment_summary}\n\n"
                f"Respond in a way that builds trust, clarifies concerns, "
                f"and maintains a cooperative tone. Keep it concise."
            )
        }
    ]
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=chat_messages,
        max_tokens=4096,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# 4. Example usage
if __name__ == "__main__":
    # Example message from a co-founder or colleague
    coworker_message = (
        "I'm feeling a bit overwhelmed by our deadline. Your suggestions "
        "are great, but I'm worried we might miss some key details."
    )

    # First call: Sentiment Analysis
    sentiment_summary = analyze_sentiment(coworker_message)
    print("=== Sentiment Analysis ===")
    print(sentiment_summary)

    # Second call: Empathy-based Response
    recommended_response = generate_empathy_response(coworker_message,
                                                     sentiment_summary)
    print("\n=== Recommended Response ===")
    print(recommended_response)