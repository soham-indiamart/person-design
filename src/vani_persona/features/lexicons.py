"""Phrase lexicons for seller utterances (English, romanized Hindi/Hinglish, Devanagari).

These are a starting point. Each list should be reviewed against real transcripts
during the data audit: precision is checked on a hand-labelled sample, and phrases
that fire on unrelated speech get removed. Matching is done on normalized text
(see ``features.transcript.normalize``) using whole-word / whole-phrase boundaries.
"""

from __future__ import annotations

# Explicit requests to repeat or clarify -> comprehension difficulty or audio issue.
CLARIFICATION = [
    "kya bola", "kya kaha", "samjha nahi", "samajh nahi", "samjhi nahi", "nahi samjha",
    "phir se", "fir se", "dobara", "repeat", "say again", "pardon", "sorry what",
    "didn't understand", "did not understand", "not clear", "clear nahi",
    # Bare "matlab" is a filler ("I mean"): 75% of clarification hits on PS07, ~1/12 genuine
    # in a spot check. Only explicit question forms count.
    "matlab?", "kya matlab", "matlab kya", "iska matlab kya",
    "what do you mean", "awaz nahi", "awaaz nahi", "sunai nahi",
    "क्या बोला", "समझ नहीं", "फिर से", "दोबारा", "क्या मतलब", "सुनाई नहीं",
]

# Explicit requests for brevity / getting to the point.
BREVITY_REQUEST = [
    "jaldi bolo", "jaldi batao", "short mein", "short me", "seedha bolo", "sidha bolo",
    "point pe aao", "point par aao", "come to the point", "get to the point",
    "be quick", "quickly", "in short", "kaam ki baat", "mudde pe",
    "जल्दी बोलो", "जल्दी बताओ", "सीधा बोलो", "पॉइंट पे", "काम की बात",
]

# Statements of being busy / lacking time right now.
TIME_PRESSURE = [
    "busy hoon", "busy hu", "busy hai", "abhi busy", "time nahi", "waqt nahi",
    "baad mein", "baad me", "later", "call back", "callback", "meeting mein",
    "driving", "gaadi chala", "dukaan pe", "customer hai", "not now", "abhi nahi",
    "व्यस्त", "बिज़ी", "टाइम नहीं", "बाद में", "अभी नहीं",
]

# Objection categories.
OBJECTION_PRICE = [
    "kitna paisa", "kitne paise", "kitna charge", "charges", "fees", "fee", "price",
    "cost", "mehenga", "mehnga", "costly", "expensive", "paisa nahi", "budget",
    "free hai", "paid", "subscription", "कितना पैसा", "महंगा", "पैसे", "चार्ज",
]
OBJECTION_TRUST = [
    "fraud", "fake", "spam", "scam", "bharosa", "bharosa nahi", "bhrosa", "trust",
    "kaise pata", "how do i know", "genuine", "real hai", "dhokha", "kaun bol",
    "who is this", "kahan se", "number kahan se", "फ्रॉड", "भरोसा", "धोखा", "कौन बोल",
]
OBJECTION_NOT_INTERESTED = [
    "interest nahi", "interested nahi", "not interested", "nahi chahiye", "zarurat nahi",
    "jarurat nahi", "need nahi", "don't need", "do not need", "mat karo", "nahi karna",
    "नहीं चाहिए", "ज़रूरत नहीं", "जरूरत नहीं", "इंटरेस्ट नहीं",
]
OBJECTION_ALREADY_HAVE = [
    "already", "pehle se", "pahle se", "ho gaya hai", "kar diya", "le liya",
    "already registered", "already done", "पहले से", "हो गया", "कर दिया",
]
INFO_REQUEST = [
    "details bhejo", "detail bhejo", "whatsapp", "message karo", "mail karo", "email karo",
    "send details", "send me", "link bhejo", "brochure", "डिटेल भेजो", "व्हाट्सएप",
]

# Opt-out. Handled by the policy layer: ends or suppresses calling, never a persona signal.
OPT_OUT = [
    "call mat karo", "phone mat karo", "dobara call mat", "don't call", "do not call",
    "stop calling", "remove my number", "number hatao", "dnd", "कॉल मत करो", "फोन मत करो",
]

# Explicit language preference statements -> immediate language override.
LANGUAGE_REQUEST = {
    "hi": ["hindi mein", "hindi me", "hindi bolo", "speak in hindi", "हिंदी में", "हिंदी बोलो"],
    "en": ["english mein", "english me", "english bolo", "speak in english", "in english", "इंग्लिश में"],
}

# Cooperation / affirmation markers.
AFFIRMATION = [
    "haan", "han", "haa", "ha ji", "haan ji", "ji haan", "ji", "theek hai", "thik hai",
    "accha", "achha", "ok", "okay", "yes", "sure", "bilkul", "zaroor", "sahi hai",
    "हाँ", "हां", "जी", "ठीक है", "अच्छा", "बिल्कुल", "ज़रूर",
]

# Question markers (in addition to a literal "?").
QUESTION_WORDS = [
    "kya", "kaise", "kitna", "kitne", "kab", "kaun", "kyun", "kyon", "kahan", "kidhar",
    "what", "how", "why", "when", "which", "where", "who", "can you", "will it", "is it",
    "क्या", "कैसे", "कितना", "कब", "कौन", "क्यों", "कहाँ",
]

# Function words used to estimate code-switching. Content words are deliberately
# excluded because romanized Hindi content words are indistinguishable from
# unknown English tokens without a full language-ID model.
HINDI_FUNCTION_WORDS = {
    "hai", "hain", "ho", "hoon", "hu", "tha", "thi", "the", "nahi", "nahin", "na", "mein",
    "me", "ka", "ki", "ke", "ko", "se", "par", "pe", "aur", "ya", "bhi", "toh", "to",
    "kya", "kaise", "kyun", "kab", "kaun", "yeh", "ye", "woh", "wo", "aap", "hum", "main",
    "mera", "meri", "mere", "aapka", "aapki", "abhi", "kuch", "sab", "bahut", "accha",
    "achha", "theek", "thik", "haan", "ji", "matlab", "lekin", "kyunki", "agar",
}
ENGLISH_FUNCTION_WORDS = {
    "the", "is", "are", "was", "were", "am", "and", "or", "but", "if", "of", "in", "on",
    "at", "for", "with", "this", "that", "it", "you", "your", "we", "our", "they", "my",
    "what", "how", "why", "when", "which", "can", "will", "would", "should", "please",
    "not", "no", "yes", "do", "does", "have", "has", "about", "from", "there", "here",
}
# Tokens present in both sets are ambiguous and are ignored for code-switch detection.
AMBIGUOUS_FUNCTION_WORDS = HINDI_FUNCTION_WORDS & ENGLISH_FUNCTION_WORDS
