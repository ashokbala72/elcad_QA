from openai import OpenAI

def check_rules(layout_data, symbol_data, api_key):
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are an expert in quality assurance of electrical drawings. "
                           "You will be given a list of text blocks and detected electrical symbols. "
                           "Analyze them and report any issues such as missing labels, misclassified symbols, "
                           "incorrect placements, or violations of naming/zone conventions."
            },
            {
                "role": "user",
                "content": f"Text blocks extracted from layout: {layout_data}\n"
                           f"Symbols detected with their positions and confidence: {symbol_data}\n"
                           f"Please return a list of abnormalities or rule violations."
            }
        ],
        max_tokens=500
    )

    return {
        "issues": [response.choices[0].message.content.strip()]
    }
