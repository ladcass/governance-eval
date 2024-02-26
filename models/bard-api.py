import pathlib
import textwrap

import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown


genai.configure(api_key='')
model = genai.GenerativeModel('gemini-pro')

prompt = 'Give me a 3 line explanation of the concept of fundamental analysis for stocks.'
print(model.generate_content(prompt).text)
