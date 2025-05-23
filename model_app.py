import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

category_descriptions = {
    'Библиотека':'Материалы об исторических событиях и жизни в прошлом, особенно в контексте военных лет, Петрограда, солдат и хронологии начала XX века.',
    'Россия':'Новости и события, связанные с внутренней политикой, обществом, заявлениями официальных лиц и жизнью в российских городах, включая Москву.',
    'Мир':'Международная повестка: события за рубежом, политика США, мировая экономика, конфликты и международные отношения.',
    'Экономика':'Вопросы, касающиеся российского и глобального бизнеса, рубля, доллара, цен, инфляции,ВВП а также деятельности компаний и рыночных трендов.',
    'Интернет и СМИ':'Темы, связанные с интернетом, СМИ, цифровыми технологиями и крупными IT-компаниями (например, Microsoft), а также новостными порталами.',
    'Спорт':'События из мира спорта: матчи, турниры, чемпионаты, сборные команды и результаты соревнований.',
    'Культура':'Кино, театр, музыка, роли актеров и режиссёров. Анонсы и обсуждения культурных событий и персоналий.',
    'Наука и техника':'Новости науки и технологий, исследования, космос (МКС, NASA), разработки и мнения учёных.',
    'Бизнес':'Экономическая активность компаний, инвестиции, отчетность, валютные потоки, изменения в бизнес-среде.',
    'Из жизни':'Лайфстайл, интересные события, происшествия, человеческие истории и нестандартные новости.',
    'Бывший СССР':'Политические и социальные события в странах бывшего Советского Союза: Украина, Молдова, Киргизия и другие',
    'Дом':'Недвижимость, строительство, стоимость жилья, квадратные метры, компании-застройщики и рынок жилья.',
    'Путешествия':'Темы туризма, поездок, путешествий внутри России и за границу, статистика и потребительское поведение туристов.',
    'Силовые структуры':'Работа МВД, судов, следственных органов, дела против чиновников, правонарушения и силовая политика.',
    'Ценности':'Мода, бренды, коллекции одежды, часы, товары класса люкс и потребительское поведение в этой сфере.',
    'Легпром':'Новости лёгкой промышленности, производства текстиля и одежды, статистика Минпромторга и экспортно-импортная динамика.',
    'Культпросвет':'Деятельность Минкультуры, общественные и культурно-просветительские инициативы, высказывания чиновников.',
    'Крым':'События в Крыму и Севастополе, деятельность местных властей, политика интеграции и региональное управление на полуострове.',
    '69-я параллель':'Жизнь и события в Мурманской области и других северных регионах, арктическая повестка, бюджет и развитие инфраструктуры.'
}

@st.cache_resource
def load_models():
    dense_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    descs = list(category_descriptions.values())
    desc_embeddings = dense_model.encode(descs, convert_to_tensor=True, normalize_embeddings=True)

    llm_model = AutoModelForCausalLM.from_pretrained(
        "t-tech/T-lite-it-1.0"
    )
    tokenizer = AutoTokenizer.from_pretrained("t-tech/T-lite-it-1.0")
    generator = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)

    return dense_model, desc_embeddings, generator

dense_model, desc_embeddings, generator = load_models()
index_to_class = list(category_descriptions.keys())
descriptions = list(category_descriptions.values())

def predict_with_llm(text, top_k=5):
    query_embedding = dense_model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(query_embedding, desc_embeddings)[0]
    top_indices = torch.topk(scores, k=top_k).indices

    candidates = [(index_to_class[i], descriptions[i]) for i in top_indices]

    prompt = f"Определи, к какой из следующих категорий относится текст:\n\n"
    prompt += f"Текст: \"{text}\"\n\nКатегории:\n"
    for i, (label, desc) in enumerate(candidates, 1):
        prompt += f"{i}. {label} — {desc}\n"
    prompt += "\nВыбери номер подходящей категории и укажи только её название.\nОтвет:"

    output = generator(prompt, max_new_tokens=20, temperature=0.3, top_p=0.9, do_sample=False, return_full_text=False)
    result = output[0]["generated_text"]

    for label, _ in candidates:
        if label in result:
            return label
    return candidates[0][0]

st.title("Классификатор текста")

user_input = st.text_area("Введите текст для классификации")

if st.button("Классифицировать"):
    if user_input.strip():
        with st.spinner("Классифицируем..."):
            category = predict_with_llm(user_input)
            st.success(f"Категория: **{category}**")
            st.write(category_descriptions[category])
    else:
        st.error("Введите текст!")
