# Lab 03 – Developing a Tiny Transformer Language Model

## 1. Laboratory Session

**Module:** PUSL3205 – Large Language Models and Generative AI  
**Laboratory:** Lab Sheet 03  
**Title:** Developing a Tiny Transformer Language Model  
**Dataset:** TinyStories  
**Framework:** PyTorch  
**Platform:** Google Colab / Jupyter Notebook

---

## 2. Objective

The main objective of this practical session was to understand the architecture and working principles of a Transformer-based language model by developing and training a small language model using PyTorch.

This practical focused on loading and preprocessing a text dataset, implementing character-level tokenization, building a Transformer architecture, training and evaluating the model, generating text, and comparing different model configurations.

It also aimed to understand how Transformer components such as self-attention, positional embeddings, and feed-forward networks contribute to language modelling.

---

## 3. Tasks and Implementation

The following tasks were completed during this practical session.

### Task 1 – Loading the Dataset

The TinyStories dataset was loaded for training the language model. A subset of 5,000 stories was used for the experiment.

### Task 2 – Exploring Story Lengths

The dataset was analysed to understand the distribution of story lengths. The number of characters and words in each story was examined using basic statistical analysis and visualisation.

### Task 3 – Character-Level Tokenization

A character-level tokenizer was implemented to convert text into numerical representations.

- Identified the unique characters in the dataset.
- Created character-to-integer and integer-to-character mappings.
- Encoded text into numerical sequences.
- Tested decoding to verify that the original text could be reconstructed.

### Task 4 – Preparing Training and Validation Data

The dataset was divided into training and validation sets using a 90:10 split.

Input and target sequences were prepared for next-character prediction, allowing the model to learn how to predict the next character based on previously observed characters.

### Task 5 – Building the Baseline Transformer Model

A baseline Transformer language model was implemented using PyTorch.

The architecture included:
- Character embeddings
- Positional embeddings
- Masked self-attention
- Multi-head attention
- Feed-forward neural networks
- Residual connections
- Layer normalisation
- Dropout

The baseline model consisted of four Transformer layers, four attention heads, and an embedding dimension of 128.

### Task 6 – Training the Transformer Model

The baseline model was trained for five epochs using the training dataset.

Training and validation losses were monitored to understand how effectively the model learned patterns from the dataset.

### Task 7 – Evaluating Model Performance

The trained model was evaluated using validation loss and perplexity.

These metrics were used to assess how well the model predicted characters that were not included in the training data.

### Task 8 – Generating Text

The trained model was used to generate stories from different starting prompts, including:

- "One day, a little girl"
- "The little dog"
- "Once upon a time"

The generated text was examined to understand the model's ability to produce meaningful story continuations.

### Task 9 – Comparing Transformer Architectures

A smaller Transformer model was implemented and compared with the baseline model.

The smaller model used two Transformer layers and two attention heads. The comparison focused on trainable parameters, validation loss, perplexity, and training time.

### Task 10 – Saving and Loading the Model

The trained baseline model was saved as a checkpoint file named `tiny_transformer_checkpoint.pth`.

The checkpoint was then loaded to reconstruct the trained model and generate text again without repeating the training process.

---

## 4. Results and Evidence

The practical produced the following results:

| Experiment | Result |
|---|---|
| Dataset | 5,000 TinyStories |
| Average story length | Approximately 835 characters |
| Average word count | Approximately 162 words |
| Shortest story | 253 characters |
| Longest story | 4,123 characters |
| Unique characters | 90 |
| Training/validation split | 90% / 10% |
| Training sequences | 29,437 |
| Validation sequences | 3,270 |

### Baseline Transformer Model

| Configuration | Value |
|---|---:|
| Transformer layers | 4 |
| Attention heads | 4 |
| Embedding dimension | 128 |
| Dropout | 0.1 |
| Trainable parameters | 821,248 |
| Initial training loss | 4.4966 |
| Initial random baseline loss | 4.4998 |
| Final validation loss | 1.1284 |
| Final perplexity | 3.09 |

### Training Performance

The baseline model showed a reduction in both training and validation losses during the five training epochs.

| Epoch | Training Loss | Validation Loss |
|---|---:|---:|
| 1 | 2.1849 | 1.6626 |
| 5 | 1.2117 | 1.1284 |

### Model Comparison

| Metric | Baseline Model | Smaller Model |
|---|---:|---:|
| Transformer layers | 4 | 2 |
| Attention heads | 4 | 2 |
| Trainable parameters | 821,248 | 424,704 |
| Validation loss | 1.1284 | 1.1994 |
| Perplexity | 3.09 | 3.32 |
| Training time per epoch | Approximately 14 seconds | Approximately 7 seconds |

The baseline model achieved better validation performance, while the smaller model required fewer parameters and less training time.

### Evidence

The completed Jupyter/Google Colab notebook contains the implementation and experimental outputs, including:

- Dataset exploration and story-length analysis.
- Character-level tokenization results.
- Training and validation data preparation.
- Transformer model architecture and parameter information.
- Training and validation loss outputs.
- Model evaluation results.
- Generated story samples.
- Model comparison results.
- Model checkpoint saving and loading.

---

## 5. Observations

The experiments demonstrated that the TinyStories dataset contained stories of different lengths and that character-level tokenization successfully converted text into numerical sequences.

The baseline Transformer model learned patterns from the training dataset, as demonstrated by the decreasing training and validation losses. The final validation loss of 1.1284 and perplexity of 3.09 indicated that the model had learned to predict characters more effectively than at initialisation.

Although the model generated text based on different prompts, the generated stories contained grammatical mistakes, incomplete sentences, and inconsistencies in meaning. This demonstrated that a reduction in prediction loss does not necessarily guarantee that generated text will be fully coherent or meaningful.

The comparison between the two model configurations showed a trade-off between model performance and computational efficiency. The smaller model had approximately half the trainable parameters and required around half the training time per epoch. However, its validation loss and perplexity were slightly higher than those of the baseline model.

Saving and reloading the trained model was also successful. This demonstrated the usefulness of model checkpoints for preserving trained parameters and reusing a model without training it again.

---

## 6. Learning Outcomes

Through this practical session, I gained practical knowledge and experience in:

- Understanding the basic architecture and working principles of Transformer language models.
- Loading and exploring a text dataset using Python.
- Applying character-level tokenization and converting text into numerical sequences.
- Preparing training and validation data for next-character prediction.
- Implementing a small Transformer model using PyTorch.
- Understanding the role of self-attention, multi-head attention, and positional embeddings.
- Training and evaluating a language model using training loss, validation loss, and perplexity.
- Generating text using a trained Transformer model.
- Comparing model configurations based on performance and computational requirements.
- Saving and loading trained model checkpoints.

---

## 7. Conclusion

This practical session provided hands-on experience in developing a small Transformer-based language model using the TinyStories dataset and PyTorch. The practical covered the complete process, from dataset preparation and character-level tokenization to Transformer implementation, model training, evaluation, text generation, and checkpoint management.

The experiments demonstrated that the baseline Transformer model could learn patterns from the dataset and generate text continuations. However, the generated stories also highlighted limitations in grammatical accuracy and contextual consistency.

Furthermore, comparing the baseline and smaller models demonstrated the relationship between model size, computational efficiency, and prediction performance.

Overall, this practical improved my understanding of Transformer architecture, attention mechanisms, language modelling, and the training and evaluation process of small language models.
