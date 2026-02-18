
import pandas as pd
from collections import Counter
import re

def clean(text):
    return re.sub(r'[^a-zA-Z\s]', '', str(text).lower())

df = pd.read_csv('customer_support_tickets.csv')

print("Unique Categories:", df['Ticket Type'].unique())

for cat in df['Ticket Type'].unique():
    print(f"\n--- {cat} ---")
    texts = df[df['Ticket Type'] == cat]['Ticket Description'].astype(str) + " " + df[df['Ticket Type'] == cat]['Ticket Subject'].astype(str)
    all_text = " ".join(texts)
    words = clean(all_text).split()
    print(Counter(words).most_common(10))
