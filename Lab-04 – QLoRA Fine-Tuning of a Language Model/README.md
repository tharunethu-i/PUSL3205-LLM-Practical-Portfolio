# Lab Sheet 4 – QLoRA Fine-Tuning of a Language Model

## 1. Laboratory Overview

This practical focuses on fine-tuning a pretrained instruction-tuned language model using Quantized Low-Rank Adaptation (QLoRA). The experiment uses a fictional training centre called BrightPath Skills Academy to develop a domain-specific student-support assistant.

The practical demonstrates how a pretrained language model can be adapted to answer questions about a specific organisation using a small question-and-answer dataset and parameter-efficient fine-tuning techniques.

## 2. Objectives

- Understand the concept of fine-tuning a pretrained language model.
- Identify and use a suitable instruction-tuned language model.
- Understand the role of GPU resources in model fine-tuning.
- Create a domain-specific question-and-answer dataset.
- Apply 4-bit quantization to reduce model memory requirements.
- Configure LoRA adapters for parameter-efficient fine-tuning.
- Fine-tune the model using the TRL SFTTrainer.
- Evaluate the fine-tuned model using training and evaluation loss.
- Compare the responses of the original and fine-tuned models.
- Save the trained LoRA adapter and tokenizer.

## 3. Tools and Technologies

The practical was implemented using:

- Python
- Google Colab
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- TRL
- BitsAndBytes
- Pandas
- Matplotlib

### Hardware

| Component | Details |
|---|---|
| GPU | NVIDIA Tesla T4 |
| GPU Memory | 14.56 GiB |
| CUDA Version | 13.0 |
| Training Precision | FP16 |

## 4. Base Language Model

The selected pretrained model was:

`Qwen/Qwen2.5-0.5B-Instruct`

This relatively small instruction-tuned language model was selected to demonstrate fine-tuning using limited GPU resources.

## 5. Dataset Description

A fictional student-support dataset was created for BrightPath Skills Academy, a training centre in Colombo, Sri Lanka.

The dataset contains 32 question-and-answer examples covering:

- Academy opening hours and location
- Available courses
- Course duration
- Class schedules and learning modes
- Student registration
- Certificate requirements
- Payment methods
- Refund policies
- Student support and contact information

### Dataset Split

| Dataset | Number of Examples |
|---|---:|
| Full dataset | 32 |
| Training dataset | 25 |
| Evaluation dataset | 7 |

The dataset was divided into training and evaluation subsets using an 80:20 split.

## 6. Implementation

### 6.1 GPU Identification

Checked GPU availability, CUDA support, GPU memory and training precision. Google Colab successfully provided a Tesla T4 GPU.

### 6.2 Model Selection

Loaded the pretrained Qwen2.5-0.5B-Instruct model using Hugging Face Transformers.

### 6.3 Dataset Creation

Created 32 conversational question-and-answer examples using a system prompt that instructed the assistant to answer using only the provided academy information and avoid inventing missing details.

The dataset was converted into a conversational Hugging Face Dataset and saved in JSON Lines format.

### 6.4 4-bit Quantization

Applied 4-bit quantization using BitsAndBytes with:

- NF4 quantization
- Double quantization
- FP16 computation

This reduced the memory required to load the base model.

### 6.5 Original Model Testing

Tested the original pretrained model with nine questions before fine-tuning. These questions covered known academy information, unseen questions, unknown information and a general-knowledge question.

### 6.6 LoRA Configuration

Configured LoRA adapters using the following settings:

| Parameter | Value |
|---|---|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Bias | None |
| Task Type | Causal Language Modelling |

The adapters were applied to selected attention and feed-forward projection layers.

### 6.7 Model Fine-Tuning

Used the TRL SFTTrainer to fine-tune the model.

| Training Parameter | Value |
|---|---:|
| Epochs | 6 |
| Training batch size | 2 |
| Evaluation batch size | 2 |
| Gradient accumulation steps | 2 |
| Learning rate | 0.0002 |
| Maximum sequence length | 512 |
| Optimizer | Paged AdamW 8-bit |

### 6.8 Model Evaluation

Evaluated the fine-tuned model using training loss, evaluation loss and mean token accuracy. A graph was generated to visualise training and evaluation loss throughout the fine-tuning process.

### 6.9 Model Comparison

Compared the original and fine-tuned models using nine test questions. The questions included paraphrased facts, unseen questions, questions about unavailable information and an out-of-domain general-knowledge question.

### 6.10 Saving the Model

Saved the trained LoRA adapter and tokenizer in the `brightpath-lora-adapter` directory for future use.

## 7. Results

### 7.1 Trainable Parameters

| Metric | Result |
|---|---:|
| Total parameters | 323,917,696 |
| Trainable parameters | 8,798,208 |
| Trainable percentage | 2.7162% |

Only a small proportion of the model parameters were trainable, demonstrating the parameter-efficient nature of LoRA.

### 7.2 Training and Evaluation Results

| Metric | Result |
|---|---:|
| Training epochs | 6 |
| Training runtime | 51.46 seconds |
| Reported training loss | 0.8398 |
| Evaluation loss | 0.3949 |
| Evaluation mean token accuracy | 91.84% |

The training-loss graph showed a considerable decrease during training, indicating that the model learned patterns from the training dataset.

### 7.3 Original and Fine-Tuned Model Comparison

The fine-tuned model demonstrated improvements in several academy-specific responses.

**Correct responses included:**
- Correctly identifying that the academy is closed on Mondays.
- Recognising that Saturday sessions are available.
- Correctly identifying debit cards as an accepted payment method.
- Recommending that students contact the academy to confirm current seat availability.
- Correctly answering the general-knowledge question about Japan.

**Incorrect or unsupported responses included:**
- Giving four weeks instead of the correct eight-week Python course duration.
- Failing to mention both certificate requirements: at least 80% attendance and completion of the final practical task.
- Inventing a Python course fee of $499.
- Giving an incorrect Saturday opening time of 10:00 AM instead of 9:00 AM.

These results show that fine-tuning improved some domain-specific responses but did not guarantee factual accuracy.

## 8. Observations

- The Tesla T4 GPU successfully supported the fine-tuning experiment using FP16 precision.
- The 32-example dataset provided a small domain-specific collection for adapting the language model.
- 4-bit NF4 quantization helped reduce the memory requirements of the pretrained model.
- LoRA allowed the model to be fine-tuned by updating only 2.7162% of its total parameters.
- The training-loss graph showed that the model learned from the training examples.
- The fine-tuned model provided more relevant responses to several BrightPath-specific questions.
- Some incorrect answers remained, demonstrating that fine-tuning does not completely eliminate hallucinations.
- The small dataset and limited evaluation examples restrict how confidently the model's general performance can be assessed.

## 9. Learning Outcomes

Through this practical, I gained experience in:

- Understanding pretrained and instruction-tuned language models.
- Working with GPU resources and CUDA-enabled environments.
- Preparing conversational datasets for language-model fine-tuning.
- Understanding 4-bit quantization and NF4.
- Applying LoRA adapters using PEFT.
- Fine-tuning language models using the TRL SFTTrainer.
- Evaluating training and evaluation loss.
- Comparing original and fine-tuned language-model responses.
- Identifying incorrect generated information and possible hallucinations.
- Saving and reusing LoRA adapters.

## 10. Conclusion

This practical demonstrated how a pretrained instruction-tuned language model can be adapted to a domain-specific student-support task using QLoRA. A dataset containing 32 question-and-answer examples was created for BrightPath Skills Academy, and the Qwen2.5-0.5B-Instruct model was fine-tuned using 4-bit quantization and LoRA adapters.

Only 2.7162% of the model parameters were trainable, demonstrating the efficiency of parameter-efficient fine-tuning. The fine-tuned model improved some academy-specific responses, although it continued to produce incorrect or unsupported information. Overall, this experiment improved my understanding of language-model fine-tuning, model evaluation and the limitations of using small datasets for domain adaptation.
