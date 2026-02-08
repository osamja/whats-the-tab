# Create custom tokenizer for GuitarPro (.gp) tabs

Steps for Custom Tokenization:

- Identify Token Types: Determine what constitutes a "token" in your data. In the context of guitar tabs, this could be individual notes, chords, or specific guitar techniques like slides, bends, etc.

- Design Rules: Decide how you will identify these tokens in the text or structured data. For example, you might decide that every note is represented by its pitch and octave, or that every chord starts and ends with certain symbols.

- Implement the Tokenizer: Write code to scan through your data and separate it into tokens based on the rules you've established.

- Test: Run your tokenizer on sample data and refine it as necessary. Make sure it can handle edge cases and is robust to slight variations in formatting.


## 1. Identify Token Types

