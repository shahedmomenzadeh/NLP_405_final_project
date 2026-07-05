### **Phase 1: Environment Setup & Data Preparation**

Before building the model, the data pipeline must be strictly configured according to the project rules.

1. 
**Library Initialization:** Import required libraries such as `torch`, `transformers` (for BERT), `seqeval` (for evaluation), `matplotlib` (for plotting), and `pytorch-crf` (for the CRF layer).
(I have already installed torch and GPU in enabled)

2. **Data Extraction:**
* Load the dataset provided in the directory (NLP-prj-data) system.


* Filter the dataset to extract **only the English language samples**.


* **Crucial constraint:** Do *not* apply standard text cleaning like stop-word removal. Preserve the raw sentence structure.




3. **Tokenization & Alignment (Layer 1 - Projection):**
* Initialize the `BertTokenizerFast` from the `bert-base-cased` or `bert-base-uncased` model.


* Tokenize the sentences. Since BERT uses subword tokenization (WordPiece), a single word might be split into multiple tokens.
* Implement an alignment function to ensure the original BIO labels map correctly to the new subword tokens (usually by assigning the original label to the first subword and an 'X' or 'PAD' to the rest, or propagating the 'I' label).


4. 
**Dataset & DataLoader:** Wrap the tokenized inputs, attention masks, and aligned labels into a PyTorch `Dataset` and batch them using a `DataLoader` for the Train, Validation, and Test sets.



---

### **Phase 2: Model Architecture (Part 1)**

The architecture must strictly follow the 4-layer structure outlined in the assignment.

1. **Layer 1: Projection Layer**
* This is inherently handled by the Tokenizer and the embedding layer of the BERT model, converting token IDs into dense vectors.




2. **Layer 2: BERT-base Contextual Representation**
* Load the pre-trained `bert-base` model.


* **Freezing Parameters:** Iterate through the named parameters of the BERT model. Disable gradient tracking (`requires_grad = False`) for the embedding layer and the first 10 encoder layers.


* Leave the 11th and 12th encoder layers active for fine-tuning.




3. **Layer 3: Neural Network Interface**
* Design two separate modules to interface between BERT and the CRF:


* **Option A (MLP):** A Multi-Layer Perceptron (e.g., Linear -> ReLU -> Dropout -> Linear).
* **Option B (Bi-LSTM):** A Bidirectional LSTM layer that processes the BERT hidden states.


* The output dimension of these modules must match the number of BIO tags in the dataset.




4. **Layer 4: CRF Layer**
* Initialize a Conditional Random Field layer using `pytorch-crf`.


* The CRF module will take the emission scores from Layer 3 and the true sequence tags during training to compute the log-likelihood loss. During inference, it will use the Viterbi algorithm to decode the most likely tag sequence.





---

### **Phase 3: Training & Hyperparameter Tuning**

1. **Optimizer & Scheduler:** Set up an optimizer (like AdamW). Use a lower learning rate for the BERT layers (e.g., 2e-5) and a slightly higher one for the custom NN and CRF layers (e.g., 1e-3).
2. **Training Loop:**
* Pass inputs and attention masks through BERT.
* Pass the output hidden states through the chosen NN layer (MLP or Bi-LSTM).
* Calculate the negative log-likelihood loss using the CRF layer.
* Backpropagate and update the un-frozen weights.


* Track the training loss per epoch.


3. **Validation Loop:** Evaluate the loss on the validation set at the end of each epoch.
4. 
**Hyperparameter Tuning:** Run training cycles iterating through different configurations of Layer 3 (e.g., varying the number of layers and hidden neurons in the MLP and Bi-LSTM) to find the optimal setup.



---

### **Phase 4: Evaluation & Part 1 Reporting**

1. 
**Loss Visualization:** Use `matplotlib` to plot the training loss vs. validation loss over the epochs.


2. **Metrics Calculation:**
* Pass the Test set through the trained model to generate predicted tag sequences.
* Use the `seqeval` library to calculate Accuracy, Precision, Recall, and F1-score.


* Ensure these metrics are calculated at the class level and as Macro/Micro averages.




3. 
**Documentation:** Compile the tuning results into a comparison table.



---

### **Phase 5: LLM Experimentation (Part 2)**

This phase evaluates whether a zero-shot/few-shot LLM can compete with a fine-tuned BERT.

1. 
**Sampling:** Write a script to perform **stratified sampling** to select exactly 20 diverse representative samples from the Test set.


2. 
**LLM Setup:** Integrate an API (e.g., OpenAI GPT, Anthropic, or a local Llama/Qwen instance via `transformers` or `Ollama`).


3. **Prompt Engineering (Zero-Shot):**
* Design a prompt that explains the sequence labeling task, lists all 80+ possible BIO classes, and provides the input sentence.


* 
**Constraint:** Send exactly one prompt per sample with no conversational memory/context from previous samples.


* Parse the LLM's text output back into a structured BIO list.


4. **Prompt Engineering (Few-Shot):**
* Modify the zero-shot prompt to include a few labeled examples.


* 
**Constraint:** Ensure these examples are pulled from the Training set, *not* the Test set.




5. 
**BERT Baseline on Subset:** Run your fine-tuned BERT model on these exact same 20 samples to get a baseline F1 score for comparison.



---

### **Phase 6: Final Analysis & Output Generation**

1. 
**Comparative Charting:** Plot a bar chart comparing the F1 scores of: BERT (on the 20 samples), LLM Zero-shot, and LLM Few-shot.


2. 
**Qualitative Analysis:** Write an analytical section examining the specific errors made by the LLM versus BERT, and hypothesize how the prompt structure affected the LLM's accuracy.


3. **Deliverables Formatting:** Ensure all executed code is saved purely as `.py` scripts. Do not submit Jupyter Notebooks (`.ipynb`). Prepare the final brief report containing the charts, tables, and analytical text.