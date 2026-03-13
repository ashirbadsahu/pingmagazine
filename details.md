# Gemma 3 Bidirectional Translation Agent Guide

## 1. Dataset Construction

To create a true "agent," the model must be trained on both translation directions simultaneously. Research shows that models trained this way can even learn to align languages better than those trained on a single direction.

### Dataset Composition (~77k Total Rows)

* **Parallel Translation:** 70,000 rows (`Dataset_English_Hindi.csv`) + 5,000 high-quality rows (`news-commentary-en-hi.csv`).
* **Monolingual Alignment:** ~700 rows (`hi-hi-paraphrase.csv` and `en-en-paraphrase.csv`). These "target-only" generation tasks are critical for reducing **Off-target** (wrong language) and **Oscillatory Hallucination** (repetition) errors.



### Bidirectional Merging Strategy

When merging your files into a single JSON for fine-tuning, you should duplicate your parallel rows to train both directions:

1. **Direction A (En $\rightarrow$ Hi):** Use English as input, Hindi as output.
2. **Direction B (Hi $\rightarrow$ En):** Flip the same row; use Hindi as input, English as output.
3. **Monolingual Pass:** Use the paraphrase files to teach the model to stay "within" one language.

### The Instruction Template

Apply the Gemma 3 chat template to distinguish the tasks.

* **English to Hindi:** `<start_of_turn>user\nTranslate to Hindi: {en_text}<end_of_turn>\n<start_of_turn>model\n{hi_text}<end_of_turn>`
* **Hindi to English:** `<start_of_turn>user\nTranslate to English: {hi_text}<end_of_turn>\n<start_of_turn>model\n{en_text}<end_of_turn>`
* **Alignment (Hi):** `<start_of_turn>user\nRewrite this in Hindi: {hi_text}<end_of_turn>\n<start_of_turn>model\n{hi_paraphrase}<end_of_turn>`
* **Alignment (En):** `<start_of_turn>user\nRewrite this in English: {en_text}<end_of_turn>\n<start_of_turn>model\n{en_paraphrase}<end_of_turn>`

---

## 2. Fine-Tuning Strategy (QLoRA)

Since **Gemma 3 270M** is small, it can be fine-tuned very efficiently. We use **QLoRA** to modify the model's capacity while keeping it lightweight.

### Hyperparameters

* 
**Rank (r):** 16. A rank of 16-32 is optimal for a 270M model to handle the complexity of bidirectional translation without losing capacity.


* **Alpha:** 32.
* **Target Modules:** `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`. Targeting all linear layers is necessary for translation tasks.


* **Learning Rate:** $2 \times 10^{-4}$. This is a stable rate for LoRA on small architectures.


* **Epochs:** 2–3. Scaling the number of training examples generally follows a log-linear improvement curve.



---

## 3. Export to .task (LiteRT)

For deployment on a **Snapdragon 695**, **INT4 quantization** is mandatory to optimize memory bandwidth and ensure smooth performance.

### Step A: Weight Merging

Combine the LoRA adapters with the base model to create a standalone translator.

```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_bidirectional_translator")

```

### Step B: INT4 Conversion

Convert the PyTorch model to a LiteRT (TFLite) flatbuffer.

```python
import ai_edge_torch
from ai_edge_torch.quantize import ptq_int4

edge_model = ai_edge_torch.convert(
    merged_model,
    sample_input, # Use a sample tensor of tokens for tracing
    quant_config=ai_edge_torch.QuantConfig(ptq_int4())
)
edge_model.export("gemma3_270m_bidirectional_translator_int4.tflite")

```

### Step C: MediaPipe Bundling

Bundle the `.tflite` model and the `tokenizer.model` into a single `.task` file using the MediaPipe LLM Bundler. This final artifact is what your mobile application will load.

