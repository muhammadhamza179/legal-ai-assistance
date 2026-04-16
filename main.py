from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "user", "content": "WHAT IS THE PUNISHEMENT OF SLAPPING SOMEONE?"}
    ]
)

print(response.choices[0].message.content)