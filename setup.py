from setuptools import find_packages, setup

setup(
    name="medical_chatbot",
    version="0.1.0",
    author="skander kolsi",
    author_email="skouza951@gmail.com",
    packages=find_packages(),
    install_requires=[]
)

"""with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="medical_chatbot",
    version="0.1.0",
    author="Skander Kolsi",
    author_email="skouza951@gmail.com",
    description="A medical chatbot using LangChain, Pinecone and Groq LLM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "pinecone-client",
        "huggingface-hub",
        "python-dotenv",
    ],
    python_requires="=3.10",
)"""