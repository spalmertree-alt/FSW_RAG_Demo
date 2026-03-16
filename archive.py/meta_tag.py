import pandas as pd

# Load your original CSV
df = pd.read_csv("ror_questions_tag.csv")

# Define tags and keywords
tags = {
    "#inland": ["inland"],
    "#international": ["international"],
    "#anchor": ["anchor"],
    "#overtaking": ["overtake", "overtaking"],
    "#lights": ["light", "lights", "flashing"],
    "#sound": ["sound", "signal"],
    "#horns": ["horn"],
    "#whistle": ["whistle", "blast"],
    "#collision": ["collision", "danger", "doubt"],
    "#passing": ["pass", "passing"],
    "#crossing": ["crossing", "cross"],
    "#channel": ["channel"],
    "#dock": ["dock"],
    "#berth": ["berth"],
    "#sail": ["sail"],
    "#sailing": ["sailing"],
    "#barge": ["barge"],
    "#head_on": ["head-on", "head on"],
    "#anchorage": ["anchorage"],
    "#tug": ["tug"],
    "#astern": ["astern"],
    "#tow": ["tow"],
    "#towed": ["towed"],
    "#fog": ["fog"]
}

def tag_question(text):
    text_lower = str(text).lower()
    applied_tags = []
    for tag, keywords in tags.items():
        if any(keyword in text_lower for keyword in keywords):
            applied_tags.append(tag)
    return " ".join(applied_tags)

# Apply tagging to every row
df["Meta_Data"] = df["Question_Text"].apply(tag_question)

# Save the enriched CSV
df.to_csv("meta_data_test.csv", index=False)
print("Tagged CSV created: ror_questions_tagged.csv")
