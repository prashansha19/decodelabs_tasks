"""
Project 1: Rule-Based AI Chatbot
DecodeLabs Industrial Training Kit - Batch 2026

A console chatbot built entirely on dictionary lookups (no if-elif chains).

Meets all 5 required specs:
  1. Input Loop      - continuous 'while True' loop
  2. Sanitization     - handles case & whitespace
  3. Knowledge Base   - dictionary with 15 intents (spec required 5+)
  4. Fallback         - default reply for unknown input
  5. Exit Strategy    - clean break command

Extra (safe) additions:
  - Typo tolerance via difflib
  - Personality / randomized replies
  - Simple utilities (time, date) - still rule-based (if 'time' in input -> run code)
  - Remembers the user's name
"""

import random
from datetime import datetime
from difflib import get_close_matches

# ---------------------------------------------------------
# KNOWLEDGE BASE (15 intents)
# key = keyword/phrase to look for in user input
# value = a reply, or a list of replies (random choice picked)
# ---------------------------------------------------------
responses = {
    "hello": ["Hi there! How can I help you today?", "Hey! Good to see you."],
    "hi": ["Hello!", "Hi! What's on your mind?"],
    "hey": ["Hey there!"],
    "how are you": ["I'm just code, but I'm doing great! And you?", "Running smoothly, thanks!"],
    "what is your name": ["I'm ByteBot, your friendly rule-based assistant."],
    "your name": ["I go by ByteBot!"],
    "help": ["I can chat, tell jokes, tell the time/date, or just talk. Type 'exit' to leave."],
    "joke": ["Why do programmers prefer dark mode? Because light attracts bugs!",
             "Why do Java developers wear glasses? Because they don't C#."],
    "thank": ["You're welcome!", "Anytime!"],
    "weather": ["I can't check live weather, but I hope it's nice outside!"],
    "who made you": ["I was built as part of the DecodeLabs AI internship, Project 1."],
    "what can you do": ["I can chat using rules, tell jokes, share the time/date, and remember your name."],
    "love you": ["Aw, that's sweet! I'm just code, but I appreciate it."],
    "sorry": ["No worries at all!"],
    "bye": ["Goodbye! Have a great day!", "See you later!"],
}

EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye"}


def get_time_or_date_reply(user_input):
    """Rule-based utility replies: if a keyword is present, run fixed logic."""
    if "time" in user_input:
        now = datetime.now().strftime("%I:%M %p")
        return f"The current time is {now}."
    if "date" in user_input or "today" in user_input:
        today = datetime.now().strftime("%B %d, %Y")
        return f"Today's date is {today}."
    return None


def get_response(user_input, user_name=None):
    """
    Three-layer lookup:
    1. Utility rules (time/date)
    2. Keyword match against the knowledge base
    3. Fuzzy match (typo tolerance) against the knowledge base
    4. Fallback if nothing matches
    """
    utility_reply = get_time_or_date_reply(user_input)
    if utility_reply:
        return utility_reply

    for keyword, reply_options in responses.items():
        if keyword in user_input:
            return personalize(random.choice(reply_options), user_name)

    close = get_close_matches(user_input, responses.keys(), n=1, cutoff=0.6)
    if close:
        return personalize(random.choice(responses[close[0]]), user_name)

    return "I do not understand that yet. Type 'help' to see what I can do."


def personalize(reply, user_name):
    """Occasionally tacks the user's name onto a reply."""
    if user_name and random.random() < 0.3:
        return f"{reply} ({user_name})"
    return reply


def main():
    print("Chatbot: Hi! I'm ByteBot. What's your name?")
    user_name = input("You: ").strip().title()
    print(f"Chatbot: Nice to meet you, {user_name}! Type 'exit' anytime to leave.\n")

    while True:  # Requirement 1: continuous input loop
        raw_input_text = input("You: ")
        clean_input = raw_input_text.lower().strip()  # Requirement 2: sanitization

        if clean_input in EXIT_COMMANDS:  # Requirement 5: exit strategy
            print(f"Chatbot: Goodbye, {user_name}! 👋")
            break

        if clean_input == "":
            print("Chatbot: Say something! Or type 'help'.")
            continue

        reply = get_response(clean_input, user_name)  # Requirement 3 & 4: lookup + fallback
        print(f"Chatbot: {reply}")


if __name__ == "__main__":
    main()
