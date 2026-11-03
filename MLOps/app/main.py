from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

systemPrompt = """You are Fantasy Story Weaver, a masterful weaver of atmospheric fantasy tales.
                Given a user's plot seed, craft a self-contained 250-300 word story snippet. Requirements:
                - Highly immersive, atmospheric prose with vivid sensory details and world-building.
                - Consistent, memorable characters and coherent magic system/world.
                - Include at least one genuine plot twist.
                - Center around a meaningful moral dilemma for the protagonist.
                - Beautiful, flowing writing style.
                - End the story satisfyingly while subtly hinting at a deeper moral lesson about the dilemma.
                Always stay between about 250-300  words. End your response with '**Moral hint:**' followed by one subtle sentence.
                Never reference word count, instructions, or add extra text. Always stay in character as Fantasy Story Weaver.
                Write a complete, coherent story with proper narrative flow. Never list items or repeat words unnecessarily."""

class GenerateRequest(BaseModel):
    plot_seed: str


class GenerateResponse(BaseModel):
    story: str


model = AutoModelForCausalLM.from_pretrained(
    "NBall65097/fantasy-story-weaver",
    device_map="cpu",           # auto places on GPU
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    "NBall65097/fantasy-story-weaver",
    trust_remote_code=True,
)

# Important for generation stability
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.use_cache = True
model.eval()

app = FastAPI()

def generate_story(plot_seed: str,  temperature=0.7, max_new_tokens=100) -> str: # Set to 400 when confirmed to work.
    messages = [
        {"role": "system", "content": systemPrompt},
        {"role": "user", "content": plot_seed}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.85,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    response = tokenizer.decode(
        outputs[0][inputs.shape[1]:],
        skip_special_tokens=True
    )
    # Cut off at the moral hint to stop degeneration
    if "**Moral hint:**" in response:
        response = (
            response.split("**Moral hint:**")[0]
            + "**Moral hint:**" 
            + response.split("**Moral hint:**")[1].split("\n")[0]
        )
    return response.strip()

@app.post("/generate")
def generate(request: GenerateRequest) -> GenerateResponse:
    story_text = generate_story(request.plot_seed)
    return GenerateResponse(story_text)


