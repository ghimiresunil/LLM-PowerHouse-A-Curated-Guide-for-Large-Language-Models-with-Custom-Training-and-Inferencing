import gradio as gr
from config.constant import LABEL_RECURSIVE, LABEL_TEXTSPLITTER
from src.utils.gradio_utils import (
    change_preset_separators,
    change_split_selection,
    chunk,
)


def create_gradio_interface(example_text):
    with gr.Blocks(
        theme=gr.themes.Soft(
            text_size="lg", font=["monospace"], primary_hue=gr.themes.colors.green
        )
    ) as demo:
        text = gr.Textbox(label="Your text 🪶", value=example_text)
        with gr.Row():
            split_selection = gr.Dropdown(
                choices=[
                    LABEL_TEXTSPLITTER,
                    LABEL_RECURSIVE,
                ],
                value=LABEL_RECURSIVE,
                label="Method to split chunks 🍞",
            )
            separators_selection = gr.Textbox(
                elem_id="textbox_id",
                value=["\n\n", "\n", " ", ""],
                info="Separators used in RecursiveCharacterTextSplitter",
                show_label=False,  # or set label to an empty string if you want to keep its space
                visible=True,
            )
            separator_preset_selection = gr.Radio(
                ["Default", "Python", "Markdown"],
                label="Choose a preset",
                info="This will apply a specific set of separators to RecursiveCharacterTextSplitter.",
                visible=True,
            )

        with gr.Row():
            length_unit_selection = gr.Dropdown(
                choices=[
                    "Character count",
                    "Token count (BERT tokens)",
                ],
                value="Character count",
                label="Length function",
                info="How should we measure our chunk lengths?",
            )
            slider_count = gr.Slider(
                50,
                500,
                value=200,
                step=1,
                label="Chunk length 📏",
                info="In the chosen unit.",
            )
            chunk_overlap = gr.Slider(
                0,
                50,
                value=10,
                step=1,
                label="Overlap between chunks",
                info="In the chosen unit.",
            )

        out = gr.HighlightedText(
            label="Output",
            show_legend=True,
            show_label=False,
            color_map={"Overlap": "#DADADA"},
        )

        split_selection.change(
            fn=change_split_selection,
            inputs=split_selection,
            outputs=[separators_selection, separator_preset_selection],
        )

        separator_preset_selection.change(
        fn=change_preset_separators,
        inputs=separator_preset_selection,
        outputs=separators_selection,
    )
        gr.on(
            [
                text.change,
                length_unit_selection.change,
                separators_selection.change,
                split_selection.change,
                slider_count.change,
                chunk_overlap.change,
            ],
            chunk,
            [
                text,
                slider_count,
                split_selection,
                separators_selection,
                length_unit_selection,
                chunk_overlap,
            ],
            outputs=out,
        )
        demo.load(chunk, inputs=[text, slider_count, split_selection, separators_selection, length_unit_selection, chunk_overlap], outputs=out)
        return demo

from config.constant import EXAMPLE_TEXT
interface = create_gradio_interface(EXAMPLE_TEXT)
interface.launch()