## Installation and Running

1. Install Python 3.12.6
  - Download and install Python 3.12.6 from the official website:
  - [https://www.python.org/downloads/release/python-3126/](https://www.python.org/downloads/release/python-3126/)

2. Clone the Project
  - Open a command prompt in the folder where you want to store the project.
  - Run the following command to clone the repository:
  ```
  git clone https://github.com/samabogbog/OpenVINO-webui.git
  ```
3. Download the Model
  - Download the model from Hugging Face:
  - [https://huggingface.co/samabogbog/Juggernaut-XL-v9-ov](https://huggingface.co/samabogbog/Juggernaut-XL-v9-ov)
  - Alternatively, use the following command to clone the model repository:
  ```
  git clone https://huggingface.co/samabogbog/Juggernaut-XL-v9-ov
  ```
  - Place the downloaded model files in the `\models` folder within the project directory.

4. Activate the virtual environment and Install the required dependencies
  ```
  cd OpenVINO-webui

  python -m venv venv

  venv\scripts\activate

  pip install -r requirements.txt
  ```
5. Run
  ```
  python webui.py
  ```
