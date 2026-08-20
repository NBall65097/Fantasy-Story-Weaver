# Fantasy Story Weaver

**Fine-tuned Phi-3.5-mini model specialised in generating short, atmospheric fantasy stories with moral dilemmas and plot twists.**

[![Model on Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/NBall65097/fantasy-story-weaver)
[![Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/NBall65097/fantasy-storyweaver-data)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

---

### Overview

Fantasy Story Weaver is an end-to-end LLM fine-tuning project that takes a small open-source model (Phi-3.5-mini) and specialises it for creative fantasy writing. Given a short plot seed, the model generates a self-contained story snippet featuring:

- Immersive, atmospheric prose
- Consistent world-building and characters
- A genuine plot twist
- A meaningful moral dilemma
- A subtle closing moral insight

The project covers the full pipeline: dataset design, efficient fine-tuning with Unsloth + QLoRA, iterative evaluation, debugging generation issues, and deployment-ready model publishing.

---

### Example Output

**Prompt:**  
*The last remaining dream-scribe in a world that has outlawed dreaming is asked to record the final dream of a dying god, which may either save or erase the concept of hope itself.*

**Generated Story (excerpt):**  
*In the heart of the Sleepless City, under a sky eternally veiled by smog, there walked Iliana—the last dream-scribe... [full story continues with twist and resolution]*

**Moral hint:** Even in the absence of dreams, the heart finds ways to beat.

More examples are available on the [Hugging Face model card](https://huggingface.co/NBall65097/fantasy-story-weaver).

---

### Technical Approach

| Component              | Details                                      |
|------------------------|----------------------------------------------|
| Base Model             | `unsloth/phi-3.5-mini-instruct-bnb-4bit`     |
| Fine-tuning Method     | QLoRA via Unsloth + Hugging Face TRL         |
| Dataset                | Custom instruction dataset (~165 examples)   |
| Training               | 3 epochs, LR 5e-5, sequence length 1024, packing enabled |
| Hardware               | Google Colab (T4) → later moved for stability|
| Optimisations          | 4-bit quantisation, gradient checkpointing, 8-bit AdamW |

**Key insight discovered during development:**  
The model performed significantly better once the requested generation length at inference was aligned with the length distribution of the training data (~250–300 words). Larger requested lengths caused frequent degeneration into repetitive or incoherent text.

---

### Project Structure

```text
├──  Fantasy_Story_Weaver.ipynb              # Full fine-tuning pipeline
└── Fantasy_Story_Weaver_Inference.ipynb    # Inference & evaluation
├──  FSTData.jsonl                           # Sample of the training data
├── app.py                                      # Gradio demo
└── README.md
```
---

### How to Use the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "NBall65097/fantasy-story-weaver",
    device_map="auto",
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("NBall65097/fantasy-story-weaver")
```
Use the recommended system prompt (available on the model card) for best results.

---

### Live Demo

A Gradio demo is available here:  
**[Fantasy Story Weaver Space](https://huggingface.co/spaces/NBall65097/fantasy-story-weaver)**

> ⚠️ **Note:** The live demo requires GPU hardware.  
> On Hugging Face’s free CPU tier the model is too slow or runs out of memory.  
> You can still run the demo locally using the `app.py` in this repository, or load the model directly from Hugging Face.

---

### Challenges & Learnings

Generation degeneration: Early outputs frequently collapsed into word-association lists. Root cause was a mismatch between training example length and the length requested in the inference system prompt.
Colab limitations: Runtime disconnects and CUDA OOM errors required careful memory management (gradient accumulation, reduced sequence length, packing) for reliable iteration.
Evaluation: Standard loss metrics were insufficient for creative quality. Qualitative evaluation with a fixed set of 15 diverse prompts proved much more informative.
Practical fine-tuning: Learned the importance of keeping training and inference distributions closely aligned, especially for style-controlled generation tasks.


### Future Improvements

Expand the dataset with more diverse high-quality examples
Add a preference-tuning stage (DPO/ORPO) to further improve coherence
Build and deploy a public Gradio Space for interactive demos
Experiment with longer-context variants and multi-turn story continuation


### Links

Merged Model: NBall65097/fantasy-story-weaver
LoRA Adapters: (link if separate)
Dataset: NBall65097/fantasy-storyweaver-data
Base Model: Microsoft Phi-3.5-mini-instruct (via Unsloth)


### Acknowledgements

Unsloth for efficient fine-tuning
Hugging Face transformers + trl
Microsoft for the excellent Phi-3.5 model family


This project demonstrates practical skills in efficient LLM fine-tuning, dataset design for creative tasks, iterative debugging of generation quality, and end-to-end model publishing.
