## Run Qwen3 TTS on Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/herom-s/Qwen3-TTS-Colab/blob/main/Qwen3_TTS_Colab.ipynb) <br>


![1](https://github.com/user-attachments/assets/e2602945-1a69-4c59-ad89-e95d96ba7858)

![2](https://github.com/user-attachments/assets/327a6564-fed1-447e-8e73-da3143c33c49)
![3](https://github.com/user-attachments/assets/a668966e-6db2-4f0a-8d4b-549670b77e2c)
![4](https://github.com/user-attachments/assets/2016b242-6fe0-4c90-9439-86c953fbc49f)



## Installation
In short for me 
```
git clone https://github.com/NeuralFalconYT/Qwen3-TTS-Colab.git
cd Qwen3-TTS-Colab
python -m venv venv
venv\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
#for my cuda 11.8 
pip uninstall -y ctranslate2
pip install ctranslate2==3.24.0 
python app.py 

#For new run make sure your are inside this folder Qwen3-TTS-Colab
venv\Scripts\activate
python app.py

# For Potato Laptop Users Use only the 0.6B model, Avoid 1.7B models, Disable Subtitle Generation
```
### Prerequisites

* **[Python 3.10](https://www.python.org/downloads/release/python-3100/) [YT Tutorial](https://www.youtube.com/watch?v=P7Q4_pqj7uc) or newer**
* **[Git](https://git-scm.com/install/windows) [YT Tutorial](https://www.youtube.com/watch?v=t2-l3WvWvqg)**
* **NVIDIA GPU**
* **[Visual Studio Community](https://apps.microsoft.com/detail/XPDCFJDKLZJLP80) [YT Tutorial](https://youtu.be/cLrk2zt4buQ)** and Download C++ Files 
* *(Optional but recommended)* **Virtual Environment** for dependency isolation

---

## Steps to Install and Run

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/NeuralFalconYT/Qwen3-TTS-Colab.git
cd Qwen3-TTS-Colab
```

---

### 2️⃣ (Optional) Create and Activate a Virtual Environment

#### Windows

```bash
python -m venv myenv
myenv\Scripts\activate
```

#### Mac / Linux

```bash
python3 -m venv myenv
source myenv/bin/activate
```

---

### 3️⃣ Install PyTorch (⚠️ IMPORTANT)

> **You MUST install PyTorch with CUDA manually first.**

#### 🔍 Check your CUDA version

```bash
nvcc --version
```

Example output:

```
release 11.8
```
- Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) and install the version compatible with your CUDA setup.:<br>

---

#### 🔧 Install PyTorch based on your CUDA version

##### CUDA 11.8

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

##### CUDA 12.1

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

##### CUDA 12.4

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

##### CUDA 12.6

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

##### CUDA 12.8

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

##### CUDA 13.0

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
```

✅ Verify GPU is detected:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

It should print:

```
True
```

---

### 4️⃣ Install Project Dependencies

```bash
pip install -r requirements.txt
```
```
pip install -U flash-attn --no-build-isolation
```
---

### 5️⃣ (Try this if **Auto Reference Voice Transcription** or **Subtitle Generation** fails in runtime)

## Fix CTranslate2

> ⚠️ **Important Note**
> `ctranslate2` is **NOT used by Qwen3-TTS directly**,

### ❌ Common Error
<img width="923" height="157" alt="image" src="https://github.com/user-attachments/assets/0a5b4d11-6fa2-49b3-be1d-0a7a4234f987" />

If auto transcription fails and you see this message in the terminal:

```
❌ An error occurred during transcription:
Library cublas64_12.dll is not found or cannot be loaded
```

This means:

👉 **Your installed `ctranslate2` version is NOT compatible with your CUDA version**


### ✅ Solution: Install a compatible CTranslate2 version

Uninstall the current version first:

```bash
pip uninstall -y ctranslate2
```

Then install **one version at a time** until it matches your system.

#####  CUDA 11.8 (GTX 16xx, RTX 20xx)
```
pip install ctranslate2==3.24.0
```
#####  CUDA 12.x (Google Colab, RTX 30 / 40)
```
pip install ctranslate2==4.5.0
```

##### Newer environments (HuggingFace Spaces, CUDA 12.1+)
```
pip install ctranslate2==4.6.0
```

> ⚠️ These are **known working examples**, not strict rules.
> The goal is to find **a version that matches your CUDA runtime**.

### 🔗 Find all available versions

You can check all official releases here:
👉 [https://pypi.org/project/ctranslate2/](https://pypi.org/project/ctranslate2/)


### 5️⃣ Install Project Dependencies
```
python app.py
```
> ⚠️ **Warning For Potato Laptop Users**
>
> * Use **only the 0.6B model**
> * **Avoid 1.7B models** (very slow / may crash)
> * **Disable Subtitle Generation** 

### 6️⃣How to delete everything
Just delete ```Qwen3-TTS-Colab``` Folder  <br>
For Model Go to this path on windows 
```C:\Users\<your_pc_user_name>\.cache\huggingface\hub```
and delete the models 






## Credit:
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)

## Disclaimer
Don't use this model to do bad things. 
