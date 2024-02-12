# Local Disaster Preparedness Chatbot

## Overview

The Local Disaster Preparedness Chatbot is designed to provide accurate, recent, and localized information to enhance the capability of communities to prepare for and respond to natural disasters. Powered by a large language model (LLM) and leveraging advanced technologies like retrieval-augmented generation (RAG) and prompt engineering, this chatbot accesses and delivers information from local disaster risk reduction agencies within a conversational interface.

## Features

- **Personalized Information:** Offers information tailored to the specific queries of users, such as nearest shelter locations, emergency procedures, and real-time alerts.
- **File Uploads:** Accepts and processes textual and PDF documents related to disaster preparedness uploaded by the system administrator or community representatives.
- **Slash Commands:** Incorporates predefined commands for quick access to common queries and instructions.

## Installation

1. **Install Python:** Ensure you have Python 3.6 or higher installed on your system. Visit [Python Downloads](https://www.python.org/downloads/) for installation guides.

2. **Set Up Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv chatbot-venv
   ```

   Activate the virtual environment:
   - On Windows:

     ```cmd
     chatbot-venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source chatbot-venv/bin/activate
     ```

3. **Install Dependencies:**

   Install the required packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API Keys:**

   Create an `.streamlit/secrets.toml` file in your project directory and add your secret keys:

   ```plaintext
   OPENAI_API_KEY='YOUR_OPENAI_API_KEY_HERE'
   ```

5. **Configure `config.yaml`:**

   Customize the `config.yaml` file to include app descriptions, command maps, and other configurations specific to your chatbot's behavior.

## Running the Chatbot

1. Navigate to your project directory in the terminal.
2. Run the command:

   ```bash
   streamlit run app.py
   ```

3. Access the chatbot interface via the URL provided by Streamlit, typically `http://localhost:8501`.

## Contributing

We appreciate contributions from the community. If you're interested in contributing, please fork the repository and submit your pull requests. For more detailed information, please refer to `CONTRIBUTING.md`.

## License

This project is licensed under the [MIT License](LICENSE).
