import gradio as gr
from collections import OrderedDict
from transformers import pipeline

def get_all_source_languages():
    """
    Returns a human-readable `dict source_languages_names:codes` 
    based on the available models.
    """
    source_languages = {
        "English": "en",
        "French": "fr",
        "Spanish": "es",
        "German": "de",
        "Italian": "it"
    }
    source_languages = OrderedDict(sorted(source_languages.items()))
    return source_languages

source_lang_dict = get_all_source_languages()
target_lang_dict = source_lang_dict  # Reutilizamos el mismo diccionario para los idiomas de destino

def translate(input_text, source, target):
    # Convertir nombres legibles a códigos abreviados
    source_code = source_lang_dict.get(source, source)  # 'en', 'fr', etc.
    target_code = target_lang_dict.get(target, target)  # 'en', 'fr', etc.
    
    try:
        model_name = f"Helsinki-NLP/opus-mt-{source_code}-{target_code}"
        pipe = pipeline("translation", model=model_name)
        translation = pipe(input_text)
        return translation[0]['translation_text'], ""
    except Exception as e:
        return "", f"Error: {str(e)}"

with gr.Blocks() as demo:
    gr.HTML("""
    <html>
        <head><style>h1 { text-align: center; }</style></head>
        <body><h1>Open Translate</h1></body>
    </html>
    """)
    with gr.Row():
        with gr.Column():
            source_language_dropdown = gr.Dropdown(
                choices=list(source_lang_dict.keys()),
                value=list(source_lang_dict.keys())[0],
                label="Source Language"
            )
            input_textbox = gr.Textbox(
                lines=5, placeholder="Enter text to translate", label="Input Text"
            )
        with gr.Column():
            target_language_dropdown = gr.Dropdown(
                choices=list(target_lang_dict.keys()),
                value="English",
                label="Target Language"
            )
            translated_textbox = gr.Textbox(lines=5, placeholder="", label="Translated Text")
    info_label = gr.HTML("")
    btn = gr.Button("Translate")
    btn.click(
        translate,
        inputs=[input_textbox, source_language_dropdown, target_language_dropdown],
        outputs=[translated_textbox, info_label]
    )

if __name__ == "__main__":
    demo.launch(auth=("iabd", "iabd"))
