<div align="center">


<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=28&pause=1000&color=00D9FF&center=true&vCenter=true&width=700&lines=Next+Word+Prediction+using+LSTM;Deep+Learning+%7C+NLP+%7C+TensorFlow" alt="Typing SVG" />
<img width="1062" height="672" alt="image" src="https://github.com/user-attachments/assets/13277f43-a17f-4cc6-8fbb-f92d72ea1f0c" />

<br/>

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Text%20Prediction-blueviolet?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<br/>

> **A sequence-based next word prediction system powered by an LSTM neural network — trained to understand context and generate intelligent word suggestions.**

</div>

---

## 📖 Table of Contents

- [About the Project](#-about-the-project)
- [Demo](#-demo)
- [How It Works](#-how-it-works)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Performance](#-performance)
- [Future Roadmap](#-future-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧠 About the Project

This project implements a **Next Word Prediction** system using an **LSTM (Long Short-Term Memory)** neural network — a specialized RNN architecture designed to capture long-range dependencies in sequential data.

The model is trained on a custom text corpus. Given an input phrase, it predicts the most probable word to follow — the same core concept behind:

| Application | Example |
|---|---|
| 📱 Mobile Keyboards | Google Keyboard, SwiftKey |
| 🤖 AI Chatbots | Auto-response suggestions |
| ✍️ Writing Assistants | Grammarly, Notion AI |
| 📝 Text Generation | Story generators, code completion |

---

## 🎬 Demo

```
Input  : "The quick brown fox"
Output : "jumps"

Input  : "She opened the"
Output : "door"

Input  : "Machine learning is"
Output : "powerful"
```

> *Predictions depend on training corpus. Results improve significantly with larger datasets.*

---

## ⚙️ How It Works

```
Raw Text → Tokenization → N-gram Sequences → Padding → LSTM Training → Prediction
```

### Step-by-step Pipeline

**1. Text Preprocessing**
- Raw text is tokenized into integer-indexed word sequences
- N-gram sequences are created from every line of the corpus
- All sequences are padded to uniform length for batch training

**2. Embedding Layer**
- Converts word indices into dense vector representations
- Captures semantic relationships between words in a continuous space

**3. LSTM Layer**
- Learns sequential patterns and context from the padded sequences
- Maintains a memory cell to retain information over long input spans

**4. Dense Output Layer**
- Softmax activation over the full vocabulary
- Outputs a probability distribution; the word with the highest probability is selected

---

## 🏗️ Model Architecture

```
┌─────────────────────────────────────┐
│          Input Layer                │
│    (padded sequence of tokens)      │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         Embedding Layer             │
│    (vocab_size → embedding_dim)     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│           LSTM Layer                │
│    (units=150, return_seq=False)    │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│          Dense Layer                │
│    (units=vocab_size, softmax)      │
└──────────────┬──────────────────────┘
               │
         Predicted Word
```

| Layer | Output Shape | Parameters |
|---|---|---|
| Embedding | (seq_len, embed_dim) | vocab × embed_dim |
| LSTM | (lstm_units,) | ~4 × lstm_units² |
| Dense | (vocab_size,) | lstm_units × vocab |

---

## 📂 Project Structure

```
Next-Word-Prediction-Model-Using-LSTM/
│
├── 📓 notebook.ipynb         # Main Jupyter notebook (preprocessing + training + prediction)
├── 📄 dataset.txt            # Training corpus
├── 🧠 model.h5               # Saved trained LSTM model
├── 🗂️  tokenizer.pkl          # Fitted tokenizer (word → index mapping)
├── 📦 requirements.txt       # Python dependencies
└── 📘 README.md              # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have **Python 3.8+** installed.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/mittal-2004/Next-Word-Prediction-Model-Using-LSTM.git
cd Next-Word-Prediction-Model-Using-LSTM

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run via Jupyter Notebook

```bash
jupyter notebook
```

Open `notebook.ipynb` and run all cells in order:

| Cell Section | Description |
|---|---|
| 📥 Data Loading | Reads `dataset.txt` |
| 🔤 Tokenization | Builds vocabulary and index maps |
| 🧩 Sequence Creation | Generates n-gram training samples |
| 🏋️ Model Training | Trains LSTM on padded sequences |
| 🔮 Prediction | Accepts user input → returns next word |

### Predict a Word (Inline)

```python
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

def predict_next_word(text, model, tokenizer, max_seq_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word

print(predict_next_word("The quick brown", model, tokenizer, max_seq_len=10))
```

---

## 📊 Performance

| Metric | Details |
|---|---|
| Architecture | Embedding → LSTM → Dense |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Prediction Type | Single next-word (argmax) |
| Context Awareness | Yes — sequence-based |

> Model accuracy scales significantly with corpus size and diversity. The current implementation is optimized for clarity and learning purposes.

---

## 🗺️ Future Roadmap

- [ ] 🌐 Deploy as a **Streamlit** or **Gradio** web app
- [ ] 🔢 Output **top-k predictions** with confidence scores
- [ ] 📚 Train on a **larger, richer corpus** (Wikipedia, books, etc.)
- [ ] ⚡ Replace LSTM with **Transformer-based models** (GPT-2, BERT)
- [ ] 🌍 Add **multi-language support**
- [ ] 🧪 Add **unit tests** for preprocessing and prediction pipeline

---

## 🤝 Contributing

Contributions are welcome and appreciated!

```bash
# Fork → Clone → Branch → Change → PR

git checkout -b feature/your-feature-name
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
```

Please ensure your code is clean, documented, and tested before submitting a pull request.

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

Made with ❤️ by [Mittal](https://github.com/mittal-2004)

⭐ **Star this repo** if you found it useful!

</div>
