import torch
import textwrap
import streamlit as st
from PIL import Image
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

add_bg_from_local('0fFW6hj.jpg')

tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
path_model = '/home/anna/Astrologer/models/gpt2_model_new.pt'
model = GPT2LMHeadModel.from_pretrained(
    'sberbank-ai/rugpt3small_based_on_gpt2',
    output_attentions = False,
    output_hidden_states = False,
)
model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))

original_title = '<p style="font-family:sans-serif; color:White; font-size: 30px;"> <b>Astrologer 🔮</b> </p>'
st.markdown(original_title, unsafe_allow_html=True)
# st.header('Astrologer 🔮')
text_includ = '<p style="font-family:sans-serif; color:White; font-size: 18px; background-color: #764ce0"> <span class="bolded">Astrologer</span> - приложение, для генерации гороскопа.\n Просто напишите свой знак зодиака ниже и приложение выдаст вам ваш прогноз на сегодня 🪄</p>'
st.markdown(text_includ, unsafe_allow_html=True)

# image = Image.open('zodiac-sign-facts.jpg')

prompt = 'Рыбы'
text_w1 = ':blue[**Ваш знак зодиака:**]'
prompt = st.text_input(text_w1, prompt)
prompt = tokenizer.encode(prompt, return_tensors='pt')
out = model.generate(
    input_ids=prompt,
    max_length=60,
    num_beams=5,
    do_sample=True,
    temperature=1,
    top_k=50,
    top_p=0.8,
    no_repeat_ngram_size=2,
    num_return_sequences=7,
    ).numpy()

text_decode = tokenizer.decode(out[0])
txt = st.text_area(':blue[**Ваш прогноз на сегодня 🔮:**]', text_decode, height=200)
# st.write('Sentiment:', run_sentiment_analysis(txt))