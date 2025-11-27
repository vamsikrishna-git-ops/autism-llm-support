from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# -----------------------------------------------------------------------------
# Model setup
# -----------------------------------------------------------------------------

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
ADAPTER_PATH = "autism_screening_mistral_final"

# Questions copied from llm_train.ipynb so that prompts match training exactly
QUESTIONS = [
    "Does your child look at you when you call his/her name?",
    "How easy is it for you to get eye contact with your child?",
    "Does your child point to indicate that s/he wants something (e.g. a toy that is out of reach)?",
    "Does your child point to share interest with you (e.g. pointing at an interesting sight)?",
    "Does your child pretend (e.g. care for dolls, talk on a toy phone)?",
    "Does your child follow where you're looking?",
    "If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them?",
    "Would you describe your child's first words as clear and meaningful?",
    "Does your child use simple gestures (e.g. wave goodbye)?",
    "Does your child stare at nothing with no apparent purpose?",
]


class ScreeningRequest(BaseModel):
    """Input schema from the web form.

    All A1–A10 answers are encoded as 1 (Yes) or 0 (No), matching the training data.
    """

    A1: int
    A2: int
    A3: int
    A4: int
    A5: int
    A6: int
    A7: int
    A8: int
    A9: int
    A10: int

    Age: int
    Sex: str  # "m" or "f"
    Jaundice: str  # "yes" / "no"
    Family_ASD: str  # "yes" / "no"


class PredictionResponse(BaseModel):
    prediction: Literal["YES", "NO", "UNKNOWN"]
    raw_output: str
    prompt: str


def build_prompt(req: ScreeningRequest) -> str:
    """Recreate the prompt format used during training in llm_train.ipynb."""

    lines = ["Patient screening info:"]

    answers = [
        req.A1,
        req.A2,
        req.A3,
        req.A4,
        req.A5,
        req.A6,
        req.A7,
        req.A8,
        req.A9,
        req.A10,
    ]

    for i, (q, ans) in enumerate(zip(QUESTIONS, answers), start=1):
        lines.append(f"{q} Answer: {ans}")

    # Demographic info
    lines.append(
        f"Age: {req.Age}, Sex: {req.Sex}, Jaundice: {req.Jaundice}, "
        f"Family history of ASD: {req.Family_ASD}"
    )

    lines.append("")
    lines.append("Does this child have autism risk? Answer YES or NO.")

    return "\n".join(lines)


# Load tokenizer and model with LoRA adapter at process startup.
# This will take some time and GPU/CPU memory the first time the server starts.

print("[startup] Loading tokenizer and base model…", flush=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

# Use 4-bit quantization, matching the training notebook configuration.
# Requires bitsandbytes to be installed and a compatible GPU.

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print("[startup] Attaching LoRA adapter from", ADAPTER_PATH, flush=True)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, device_map="auto")
model.eval()


def generate_prediction(full_text: str) -> str:
    """Run the fine-tuned model and return the decoded continuation text."""

    # Tokenize and move tensors to the same device as the model weights
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Strip the input prompt and decode only the generated part
    gen_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return decoded


def classify_yes_no(raw: str) -> str:
    """Map the raw text to a simple YES/NO/UNKNOWN label."""

    upper = raw.upper()
    if "YES" in upper:
        return "YES"
    if "NO" in upper:
        return "NO"
    return "UNKNOWN"


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(title="Autism Screening LLM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: ScreeningRequest) -> PredictionResponse:
    """Accept questionnaire answers and return the model's autism risk prediction."""

    prompt = build_prompt(req)

    # Match the training text format exactly
    full_text = f"### Instruction:\n{prompt}\n\n### Response:\n"

    raw_output = generate_prediction(full_text)
    label = classify_yes_no(raw_output)

    return PredictionResponse(
        prediction=label,
        raw_output=raw_output,
        prompt=prompt,
    )
