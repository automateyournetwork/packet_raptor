import os
import umap.umap_ as umap
import pandas as pd
from sklearn.mixture import GaussianMixture
import json
from langchain.load import dumps, loads
import subprocess
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_experimental.text_splitter import SemanticChunker
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Message classes
class Message:
    def __init__(self, content):
        self.content = content

class HumanMessage(Message):
    """Represents a message from the user."""
    pass

class AIMessage(Message):
    """Represents a message from the AI."""
    pass

# Define a class for chatting with pcap data
class ChatWithPCAP:
    def __init__(self, json_path, token_limit=200):
        self.document_cluster_mapping = {}
        self.json_path = json_path
        self.token_limit = token_limit
        self.conversation_history = []
        self.load_json()
        self.split_into_chunks()        
        self.embed_texts()
        self.store_in_chroma()        
        self.reduce_dimensions()
        self.cluster_embeddings()
        self.setup_conversation_memory()
        self.setup_conversation_retrieval_chain()

    def load_json(self):
        self.loader = JSONLoader(
            file_path=self.json_path,
            jq_schema=".[] | ._source.layers",
            text_content=False
        )
        self.pages = self.loader.load_and_split()

    def split_into_chunks(self):
        self.text_splitter = SemanticChunker(OpenAIEmbeddings())
        self.docs = self.text_splitter.split_documents(self.pages)

    def embed_texts(self):
        self.embedding_model = OpenAIEmbeddings()
        self.embeddings = []

        for idx, doc in enumerate(self.docs):
            try:
                # Directly access the page_content attribute of the Document object
                json_content_str = doc.page_content  # Adjusted based on the assumption that doc has a .page_content attribute

                # Attempt to embed the extracted content
                embedding = self.embedding_model.embed_query(json_content_str)
                if embedding:  # Assuming embedding successfully returns a non-None result
                    self.embeddings.append(embedding)
                else:
                    st.write(f"No embedding generated for document {idx}")
            except Exception as e:
                st.write(f"Error embedding document {idx}: {e}")

        if len(self.embeddings) == 0:
            st.write("Warning: No embeddings were generated.")
        else:
            st.write("Embeddings length:", len(self.embeddings))

    def store_in_chroma(self):
        self.vectordb = Chroma.from_documents(self.docs, embedding=self.embedding_model)
        self.vectordb.persist()

    def reduce_dimensions(self, dim=2, metric="cosine"):
        self.embeddings_reduced = []  # Default to an empty list
        n_embeddings = len(self.embeddings)
        if n_embeddings > 1:
            n_neighbors = max(min(n_embeddings - 1, int((n_embeddings - 1) ** 0.5)), 2)
            try:
                self.embeddings_reduced = umap.UMAP(
                    n_neighbors=n_neighbors, n_components=dim, metric=metric
                ).fit_transform(self.embeddings)
            except Exception as e:
                st.write(f"Error during UMAP dimensionality reduction: {e}")
        else:
            st.write("Not enough embeddings for dimensionality reduction.")

    def cluster_embeddings(self, random_state=0):
        if self.embeddings_reduced is not None and self.embeddings_reduced.ndim == 2 and len(self.embeddings_reduced) > 1:
            n_clusters = self.get_optimal_clusters(self.embeddings_reduced)
            gm = GaussianMixture(n_components=n_clusters, random_state=random_state).fit(self.embeddings_reduced)
            cluster_labels = gm.predict(self.embeddings_reduced)

            # Populate the document_cluster_mapping
            for i, cluster_label in enumerate(cluster_labels):
                # If documents do not have a unique identifier, use their index
                self.document_cluster_mapping[i] = cluster_label

            st.write(f"Clustering completed with {n_clusters} clusters.")
        else:
            st.write("Reduced embeddings are not available or in incorrect shape for clustering.")

    def get_optimal_clusters(self, embeddings, max_clusters=50, random_state=1234):
        if embeddings is None or len(embeddings) == 0:
            st.write("No reduced embeddings available for clustering.")
            return 1  # Return a default or sensible value for your application
        max_clusters = min(max_clusters, len(embeddings))
        bics = [
            GaussianMixture(n_components=n, random_state=random_state)
            .fit(embeddings)
            .bic(embeddings)
            for n in range(1, max_clusters + 1)
        ]
        return bics.index(min(bics)) + 1

    def setup_conversation_memory(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def setup_conversation_retrieval_chain(self):
        self.llm = ChatOpenAI(temperature=0.7, model="gpt-4-1106-preview")
        self.qa = ConversationalRetrievalChain.from_llm(self.llm, self.vectordb.as_retriever(search_kwargs={"k": 10}), memory=self.memory)

    def retrieve_relevant_documents(self, question, top_k=10):
        """
        Retrieve top_k documents relevant to the question.
        """
        try:
            search_results = self.vectordb.search(query=question, search_type='similarity')
            return [result for result in search_results]
        except Exception as e:
            st.write(f"Error retrieving documents: {e}")
            return []

    def identify_relevant_clusters(self, documents):
        """
        Identify clusters relevant to the given list of documents.

        Parameters:
            documents (list): A list of documents for which to identify relevant clusters.

        Returns:
            set: A set of unique cluster IDs relevant to the given documents.
        """
        cluster_ids = set()
        for i, doc in enumerate(documents):
            cluster_id = self.document_cluster_mapping.get(i)
            if cluster_id is not None:
                cluster_ids.add(cluster_id)
        return cluster_ids

    def chat(self, question):
        # New: Dynamically generate priming text based on the question
        priming_text = self.generate_dynamic_priming_text(question)

        # Combine the original question with the dynamic priming text
        primed_question = priming_text + "\n\n" + question
        response = self.qa.invoke(primed_question)
        self.conversation_history.append({"You": question, "AI": response})  # New: Update conversation history dynamically
        return response

    def generate_dynamic_priming_text(self, question):
        """
        Generate dynamic priming text based on the question by summarizing relevant clusters.
        This replaces the static approach with one that adapts to the user's query.
        """
        # New: Preliminary retrieval to find relevant clusters based on the question
        relevant_docs = self.retrieve_relevant_documents(question)
        relevant_clusters = self.identify_relevant_clusters(relevant_docs)
        summaries = self.generate_summaries(relevant_clusters)  # New: Generate summaries for relevant clusters only
        
        # Combine summaries into a single priming text
        return " ".join(summaries.values())

    def prepare_clustered_data(self, clusters=None):
        """
        Prepare data for summarization by clustering.
        Can now filter by specific clusters if provided, enhancing dynamic use.
        """
        # Initialize a list for filtered documents
        filtered_docs = []

        # If specific clusters are provided, filter documents belonging to those clusters
        if clusters is not None:
            for i, doc in enumerate(self.docs):
                # Retrieve the cluster ID from the mapping using the document's index
                cluster_id = self.document_cluster_mapping.get(i)
                # If the document's cluster ID is in the specified clusters, include the document
                if cluster_id in clusters:
                    filtered_docs.append(doc)
        else:
            filtered_docs = self.docs

        # Construct a DataFrame from the filtered documents
        # You'll need to adjust how you access the text and cluster_id based on your Document object structure
        df = pd.DataFrame({
            "Text": [doc.page_content for doc in filtered_docs],  # Adjust based on your document's structure
            "Cluster": [self.document_cluster_mapping[i] for i in range(len(filtered_docs))]  # Access cluster IDs from the mapping
        })
        self.clustered_texts = self.format_cluster_texts(df)

    def format_cluster_texts(self, df):
        """Organize texts by their clusters."""
        clustered_texts = {}
        for cluster in df["Cluster"].unique():
            cluster_texts = df[df["Cluster"] == cluster]["Text"].tolist()
            clustered_texts[cluster] = " --- ".join(cluster_texts)
        return clustered_texts

    # New: Override to accept specific clusters for targeted summarization
    def generate_summaries(self, clusters=None):
        self.prepare_clustered_data(clusters)
        summaries = {}
        for cluster, text in self.clustered_texts.items():
            summary = self.invoke_summary_generation(text)
            summaries[cluster] = summary
        return summaries

    def invoke_summary_generation(self, text):
        """Invoke the language model to generate a summary for the given text."""
        template = "You are an assistant to create a detailed summary of the text input provided.\nText:\n{text}"
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        summary = chain.invoke({"text": text})
        return summary

# Function to convert pcap to JSON
def pcap_to_json(pcap_path, json_path):
    command = f'tshark -nlr {pcap_path} -T json > {json_path}'
    subprocess.run(command, shell=True)

# Streamlit UI for uploading and converting pcap file
def upload_and_convert_pcap():
    st.title('Packet Buddy - Chat with Packet Captures')
    uploaded_file = st.file_uploader("Choose a PCAP file", type="pcap")
    if uploaded_file:
        if not os.path.exists('temp'):
            os.makedirs('temp')
        pcap_path = os.path.join("temp", uploaded_file.name)
        json_path = pcap_path + ".json"
        with open(pcap_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        pcap_to_json(pcap_path, json_path)
        st.session_state['json_path'] = json_path
        st.success("PCAP file uploaded and converted to JSON.")
        st.button("Proceed to Chat", on_click=lambda: st.session_state.update({"page": 2}))

# Streamlit UI for chat interface
def chat_interface():
    st.title('Packet Buddy - Chat with Packet Captures')
    json_path = st.session_state.get('json_path')
    if not json_path or not os.path.exists(json_path):
        st.error("PCAP file missing or not converted. Please go back and upload a PCAP file.")
        return

    if 'chat_instance' not in st.session_state:
        st.session_state['chat_instance'] = ChatWithPCAP(json_path=json_path)

    user_input = st.text_input("Ask a question about the PCAP data:")
    if user_input and st.button("Send"):
        with st.spinner('Thinking...'):
            response = st.session_state['chat_instance'].chat(user_input)
            st.markdown("**Synthesized Answer:**")
            if isinstance(response, dict) and 'answer' in response:
                st.markdown(response['answer'])
            else:
                st.markdown("No specific answer found.")

            st.markdown("**Chat History:**")
            for message in st.session_state['chat_instance'].conversation_history:
                # Check if message is of type HumanMessage or AIMessage
                if isinstance(message, (HumanMessage, AIMessage)):
                    prefix = "*You:* " if isinstance(message, HumanMessage) else "*AI:* "
                    st.markdown(f"{prefix}{message.content}")
                else:
                    # If the message is a dictionary, access the content using the key
                    prefix = "*You:* " if "You" in message else "*AI:* "
                    st.markdown(f"{prefix}{message['content']}")

if __name__ == "__main__":
    if 'page' not in st.session_state:
        st.session_state['page'] = 1

    if st.session_state['page'] == 1:
        upload_and_convert_pcap()
    elif st.session_state['page'] == 2:
        chat_interface()
