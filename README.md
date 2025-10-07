# AZURE Assistant MCP

Multi-agent AI assistant powered by Azure OpenAI, Cosmos DB, and Streamlit.

## Features

- **Multi-Agent Orchestration**: Supervisor, Tool Agent, Writer, Engineer, and Validator working together
- **General Assistant Mode**: Fast, single-shot responses with GPT-4.1 or O3-mini
- **Knowledge Base Integration**: Azure Cosmos DB for document storage and retrieval
- **Document Processing**: PDF, DOCX, XLSX, and CSV ingestion with intelligent chunking
- **Scratchpad System**: Collaborative workspace for agents to build complex documents
- **Voice Input**: Azure Speech Services integration for audio transcription

## Installation

```bash
# Clone the repository
git clone https://github.com/jadericdawson/AZURE_Assistant_mcp.git
cd AZURE_Assistant_mcp

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (create .env file)
# See .env.example for required variables

# Run the application
streamlit run app_mcp.py
```

## Requirements

- Python 3.11+
- Azure OpenAI API access
- Azure Cosmos DB account
- Azure Blob Storage (optional, for file uploads)
- Azure Speech Services (optional, for voice input)

## Configuration

Create a `.env` file with your Azure credentials:

```
COSMOS_ENDPOINT=https://your-cosmos-account.documents.azure.com:443/
COSMOS_KEY=your-cosmos-key
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_KEY=your-openai-key
GPT41_DEPLOYMENT=gpt-4-1-turbo
O3_DEPLOYMENT=o3-mini
```

## Usage

### General Assistant
- Fast Q&A with knowledge base search
- Model selection (GPT-4.1 or O3-mini)
- Automatic intent detection for searches

### Multi-Agent Document Orchestrator
- Complex document creation
- Research and analysis workflows
- Collaborative agent scratchpads

## Architecture

- **Frontend**: Streamlit web interface
- **Backend**: Python with Azure services
- **Database**: Azure Cosmos DB (NoSQL)
- **AI Models**: Azure OpenAI (GPT-4.1, O3-mini)
- **Storage**: Azure Blob Storage

## License

MIT License
