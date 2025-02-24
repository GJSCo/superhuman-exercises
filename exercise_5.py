import openai
import os

# 1. Configure your API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def journaling_prompt():
    """
    This function prompts the user for their daily reflection or journaling input.
    """
    user_input = input(("Please reflect on your day. What challenges, emotions, "
               "or worries are on your mind?\n> "))
    return user_input

def generate_summary_and_reframe(user_input):
    """
    This function sends the user's journaling input to the LLM to summarize
    the key challenges and reframe them in a constructive manner.
    """
    chat_messages=[
        {
            "role": "system",
            "content": (
                "You are a helpful and emotionally supportive assistant. "
                "Please summarize the user's reflections and provide a compassionate "
                "reframing of their challenges."
            )
        },
        {
            "role": "user",
            "content": user_input
        }
    ]
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=chat_messages,
        max_tokens=4096,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def generate_next_steps_and_encouragement(summary):
    """
    This function takes the summarized and reframed content and asks the LLM
    to provide practical next steps and an encouraging message.
    """
    chat_messages=[
        {
            "role": "system",
            "content": (
                "You are a coaching assistant. Please offer 2-3 practical next steps "
                "for coping with or resolving the challenges described. Then, provide "
                "a concise, uplifting message."
            )
        },
        {
            "role": "user",
            "content": summary
        }
    ]
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=chat_messages,
        max_tokens=4096,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def main():
    # Step 1: Gather user input
    user_input = journaling_prompt()

    # Step 2: Summarize and reframe
    summary = generate_summary_and_reframe(user_input)

    # Step 3: Generate next steps and encouragement
    final_output = generate_next_steps_and_encouragement(summary)

    print("\n=== Summary & Reframe ===")
    print(summary)
    print("\n=== Next Steps & Encouragement ===")
    print(final_output)

if __name__ == "__main__":
    main()
