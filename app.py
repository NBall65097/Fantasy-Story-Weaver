import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
<<<<<<< HEAD
import spaces
=======
>>>>>>> 4a27aa1d190647b485aacc4d17032832a63a6f6a

# -----------------------------
# Model Loading
# -----------------------------
MODEL_ID = "NBall65097/fantasy-story-weaver"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
print("Model loaded!")

# -----------------------------
# System Prompt (aligned with training length)
# -----------------------------
SYSTEM_PROMPT = """You are Fantasy Story Weaver, a masterful weaver of atmospheric fantasy tales.
Given a user's plot seed, craft a self-contained story snippet. Requirements:
- Highly immersive, atmospheric prose with vivid sensory details and world-building.
- Consistent, memorable characters and coherent magic system/world.
- Include at least one genuine plot twist.
- Center around a meaningful moral dilemma for the protagonist.
- Beautiful, flowing writing style.
- End the story satisfyingly while subtly hinting at a deeper moral lesson about the dilemma.
Always stay between about 250-300 words. End your response with '**Moral hint:**' followed by one subtle sentence.
Never reference word count, instructions, or add extra text. Always stay in character as Fantasy Story Weaver. Write a complete, coherent story with proper narrative flow. Never list items or repeat words unnecessarily."""

# -----------------------------
# Generation Function
# -----------------------------
<<<<<<< HEAD
@spaces.GPU
=======
>>>>>>> 4a27aa1d190647b485aacc4d17032832a63a6f6a
def generate_story(plot_seed, temperature=0.7, max_new_tokens=450):
    if not plot_seed.strip():
        return "Please enter a plot seed."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": plot_seed.strip()}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # Streaming setup
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text

# -----------------------------
# Gradio Interface
# -----------------------------
example_prompts = [
    "A lighthouse keeper on a floating island discovers that the beam of her lantern can temporarily rewind time for anything it touches, but every use shortens the lifespan of the island itself.",
    "The last remaining dream-scribe in a world that has outlawed dreaming is asked to record the final dream of a dying god, which may either save or erase the concept of hope itself.",
    "A bridge-builder in the Realm of Fractured Skies can construct crossings between broken floating continents, but the next bridge she is commissioned to build will connect two realms that have been at war for a thousand years.",
    "In the Library of Borrowed Voices, a librarian can lend her voice to silent books so they can speak their stories aloud, but the next book demands to use her voice permanently.",
]

<<<<<<< HEAD
with gr.Blocks() as demo:
=======
with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple")) as demo:
>>>>>>> 4a27aa1d190647b485aacc4d17032832a63a6f6a
    gr.Markdown(
        """
        # 🪄 Fantasy Story Weaver
        *Fine-tuned Phi-3.5-mini specialised in atmospheric fantasy stories with moral dilemmas and plot twists.*
        
        Enter a short plot seed below and the model will weave a self-contained story.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            plot_input = gr.Textbox(
                label="Plot Seed",
                placeholder="e.g. A cartographer discovers that her maps can rewrite reality, but every change erases one of her own memories...",
                lines=4
            )
            with gr.Row():
                temperature = gr.Slider(0.3, 1.1, value=0.7, step=0.05, label="Temperature")
                max_tokens = gr.Slider(250, 600, value=450, step=25, label="Max New Tokens")
            
            generate_btn = gr.Button("Generate Story", variant="primary")

        with gr.Column(scale=3):
            output = gr.Textbox(
                label="Generated Story",
                lines=18,
<<<<<<< HEAD
=======
                show_copy_button=True
>>>>>>> 4a27aa1d190647b485aacc4d17032832a63a6f6a
            )

    gr.Examples(
        examples=example_prompts,
        inputs=plot_input,
        label="Try one of these prompts"
    )

    generate_btn.click(
        fn=generate_story,
        inputs=[plot_input, temperature, max_tokens],
        outputs=output
    )

    gr.Markdown(
        """
        ---
        **Model**: [NBall65097/fantasy-story-weaver](https://huggingface.co/NBall65097/fantasy-story-weaver)  
        Fine-tuned with Unsloth + QLoRA on a custom fantasy storytelling dataset.
        """
    )

if __name__ == "__main__":
<<<<<<< HEAD
    demo.launch(theme=gr.themes.Soft(primary_hue="purple"))
=======
    demo.launch()
>>>>>>> 4a27aa1d190647b485aacc4d17032832a63a6f6a
