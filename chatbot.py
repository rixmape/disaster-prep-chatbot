# fmt: off

import json
import os
import time
import yaml
from operator import itemgetter

import firebase_admin
import streamlit as st
from firebase_admin import credentials, firestore, storage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from document_manager import DocumentManager

# fmt: on


class ChatbotPipeline:
    def __init__(self, config, chatbot_instruction, document_manager):
        self.config = config
        self.chatbot_instruction = chatbot_instruction
        self.document_manager = document_manager
        self.pipeline = self.initialize_pipeline()

    def format_documents(self, documents):
        context = "\n\n".join(
            [
                f'Document {i}:\n\n"""\n{doc.page_content}\n"""'
                for i, doc in enumerate(documents, start=1)
            ]
        )
        return f"Use the following documents to answer the query.\n\n{context}"

    def process_input(self, input):
        """Process user input to generate context-aware prompts."""
        if not input["history"]:
            return input["question"]

        # FIX: Can't access session state. Hardcoding the prompt for now.
        prompt = (
            "Given a chat history and the latest user question which might"
            " reference context in the chat history, formulate a standalone"
            " question which can be understood without the chat history. Do"
            " not answer the question, just reformulate only if needed."
        )
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        return prompt_template | ChatOpenAI() | StrOutputParser()

    def initialize_pipeline(self):
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", f"{self.chatbot_instruction}\n\n{{context}}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        rag_chain_from_documents = (
            RunnablePassthrough.assign(
                context=(lambda docs: self.format_documents(docs["context"]))
            )
            | prompt_template
            | ChatOpenAI()
            | StrOutputParser()
        )
        return RunnableParallel(
            {
                "question": itemgetter("question"),
                "history": itemgetter("history"),
                "context": self.process_input | self.document_manager.retriever,
            }
        ).assign(answer=rag_chain_from_documents)

    def invoke(self, data):
        return self.pipeline.invoke(data)


class ChatbotAgent:
    def __init__(self):
        self.load_env_variables()
        self.load_config()
        self.initialize_firebase_services()

    def load_env_variables(self):
        """Load necessary environment variables for the application."""
        os.environ["LANGCHAIN_TRACING_V2"] = st.secrets.get(
            "LANGCHAIN_TRACING_V2",
            "false",
        )
        os.environ["LANGCHAIN_ENDPOINT"] = st.secrets.get(
            "LANGCHAIN_ENDPOINT",
            "https://api.langchain.com",
        )
        os.environ["LANGCHAIN_API_KEY"] = st.secrets.get(
            "LANGCHAIN_API_KEY",
            "",
        )
        os.environ["LANGCHAIN_PROJECT"] = st.secrets.get(
            "LANGCHAIN_PROJECT",
            "default",
        )
        os.environ["OPENAI_API_KEY"] = st.secrets.get(
            "OPENAI_API_KEY",
            "",
        )

    def load_config(self, config_file_path="config.yaml"):
        """Load the configuration file for the application."""
        with open(config_file_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        self.app_description = self.config["descriptions"]["app"]
        self.feedback_description = self.config["descriptions"]["feedback"]
        self.personas = self.config["personas"]
        self.initial_message = self.config["prompts"]["initial"].strip()

    def initialize_firebase_services(self):
        """Initialize Firebase services."""
        if not firebase_admin._apps:
            firebase_auth = dict(st.secrets["FIREBASE_AUTH"])
            firebase_credentials = credentials.Certificate(firebase_auth)
            firebase_admin.initialize_app(
                firebase_credentials,
                {"storageBucket": "streamlit-chatbot-6ee28.appspot.com"},
            )

    def configure_persona(self, custom_trait_key="custom"):
        """Display UI elements for persona selection and set the selected persona."""
        st.write(self.app_description)
        st.subheader("How should I answer your questions?")

        column1, column2 = st.columns([0.4, 0.6])
        chosen_persona = None

        with column2:
            custom_persona = {
                "traits": custom_trait_key,
                "image": "images/custom.png",
            }

            chosen_persona = st.selectbox(
                label="Select the trait of the chatbot:",
                options=self.personas + [custom_persona],
                format_func=lambda x: x["traits"].capitalize(),
            )

            with st.container(border=True, height=215):
                st.markdown("**Instruction:**")
                if custom_trait_key == chosen_persona["traits"]:
                    self.persona_instruction = st.text_area(
                        label="Chatbot Persona Instruction:",
                        placeholder="Enter a custom instruction for the chatbot persona...",
                        height=120,
                        label_visibility="collapsed",
                    )
                else:
                    self.persona_instruction = chosen_persona[
                        "instruction"
                    ].strip()
                    st.markdown(self.persona_instruction)

        with column1, st.container(border=True, height=300):
            st.image(chosen_persona["image"], use_column_width=True)

    def initialize_chat(self):
        """Set up the chatbot's components and pipelines."""
        st.write("💬 Initializing chat history...")
        self.chat_history = StreamlitChatMessageHistory()
        if not self.chat_history.messages:
            self.chat_history.add_ai_message(self.initial_message)

        st.write("📢 Connecting to user feedback database...")
        firestore_client = firestore.client()
        self.feedback_database = firestore_client.collection("feedback")

        st.write("🔍 Setting up document manager...")
        document_manager = DocumentManager(self.config)

        st.write("🔗 Setting up chatbot pipeline...")
        chatbot_instruction = (
            self.config["prompts"]["main_instruction"]
            + self.persona_instruction
        )
        self.chatbot_pipeline = ChatbotPipeline(
            self.config,
            chatbot_instruction,
            document_manager,
        )

        st.write("✨ Finishing chatbot configuration...")

    def format_documents(self, documents):
        """Format documents for the chatbot pipeline."""
        context = "\n\n".join(
            [
                f'Document {i}:\n\n"""\n{doc.page_content}\n"""'
                for i, doc in enumerate(documents, start=1)
            ]
        )
        return f"Use the following documents to answer the query.\n\n{context}"

    def configure_chat(self):
        """Configure the Streamlit UI components for the chat interface."""
        for message in self.chat_history.messages:
            st.chat_message(message.type).write(message.content)

        if user_input := st.chat_input():
            interpreted_input = self.interpret_slash_command(user_input)
            st.chat_message("human").write(interpreted_input)
            response = self.chatbot_pipeline.invoke(
                {
                    "question": interpreted_input,
                    "history": self.chat_history.messages,
                }
            )

            ai_message = st.chat_message("ai")
            ai_message.write(response.get("answer"))

            file_citations = response.get("context")
            citations_container = ai_message.expander(
                f"File Citations ({len(file_citations)}):",
                expanded=False,
            )
            for citation in file_citations:
                source = citation.metadata.get("source")
                content = citation.page_content.replace("#", "")
                content = "\n".join(
                    [f"> {line}" for line in content.split("\n")]
                )
                citations_container.markdown(f"**{source}**\n{content}")

            self.chat_history.add_user_message(interpreted_input)
            self.chat_history.add_ai_message(response.get("answer"))

    def interpret_slash_command(self, prompt):
        valid_commands = self.config["commands"]
        if prompt.startswith("/"):
            command, argument = prompt.split(" ", 1)
            command = command.replace("/", "")
            if not command in valid_commands.keys():
                return prompt
            interpreted_prompt = (
                f"{valid_commands[command]['description']}\n\n"
                f"Query: {argument}"
            )
            return interpreted_prompt
        return prompt

    def display_sidebar(self):
        """Manage sidebar content with helpful information and feedback form."""
        with st.sidebar:
            self.display_helpful_info()
            st.divider()
            self.display_feedback_form()

    def display_helpful_info(self):
        st.title("💡 Helpful Information")
        st.write(self.config["descriptions"]["app"])

        with st.expander("Predefined commands"):
            commands = self.config["commands"]
            for command_name, command_info in commands.items():
                st.markdown(
                    f":green[**{command_name}**] : "
                    f"{command_info['description']}\n\n"
                    "Sample usage:\n\n"
                    f"\t/{command_name} {command_info['arg']}\n\n"
                )

        chat_history_json = self.convert_chat_history_to_json()
        filename = f"conversation_{int(time.time())}.json"
        st.download_button(
            label="Download Conversation as JSON",
            data=chat_history_json,
            file_name=filename,
            mime="application/json",
            use_container_width=True,
            type="primary",
        )

    def convert_chat_history_to_json(self):
        return json.dumps(
            [
                {"type": message.type, "content": message.content}
                for message in self.chat_history.messages
            ],
            indent=4,
        )

    def display_feedback_form(self):
        st.title("📢 Feedback")
        st.write(self.feedback_description)

        feedback = dict(subject="", content="", history="")
        feedback["subject"] = st.selectbox(
            "Subject",
            options=[
                "💭 General feedback",
                "🌟 Feature request",
                "🚨 Bug report",
                "📢 Other",
            ],
        )
        feedback["content"] = st.text_area("User Feedback", height=100)
        include_chat_history = st.checkbox("Include chat history in feedback")
        if include_chat_history:
            feedback["history"] = self.convert_chat_history_to_json()

        if st.button("Submit", type="primary"):
            if feedback:
                feedback["timestamp"] = firestore.SERVER_TIMESTAMP
                self.feedback_database.add(feedback)
                st.success("Feedback submitted successfully!", icon="🚀")
            else:
                st.error("Give feedback before submitting.", icon="🙀")

    def run(self):
        """Main execution logic to run the chatbot application."""
        st.session_state.setdefault("is_configured", False)

        if not st.session_state.is_configured:
            self.configure_persona()

            if st.button("Start chatting!"):
                with st.status("Initializing chatbot...", expanded=True):
                    self.initialize_chat()
                st.session_state.is_configured = True
                st.rerun()
        else:
            self.configure_chat()
            self.display_sidebar()
