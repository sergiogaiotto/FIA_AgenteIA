"""
FastAPI Application - Agentes de IA
Version: 1.5.0 - Simplified
"""

import os
import asyncio
import json
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Imports dos agentes
from agents.mcp_agent import MCPAgent
from agents.workflow_agent import WorkflowAgent
from agents.rag_agent import RAGAgent
from agents.externo_agent import ExternoAgent
from agents.tool_mermaid_agent import ToolMermaidAgent
# from agents.classifica_imagem_agent import ClassificaImagemAgent  # Comentado se não existir

# Configuração de templates
templates = Jinja2Templates(directory="pages")

# Variáveis globais para agentes
agents_dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerenciador de ciclo de vida da aplicação"""
    # Startup
    print("🚀 Iniciando Agentes de IA...")
    await initialize_agents()
    yield
    # Shutdown
    print("👋 Encerrando aplicação...")


# Criar aplicação FastAPI
app = FastAPI(
    title="Agentes de IA - FIA",
    description="Plataforma com agentes especializados",
    version="1.5.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especifique os domínios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar arquivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")


# Modelos Pydantic
class ChatRequest(BaseModel):
    """Modelo para requisições de chat"""
    message: str = Field(..., min_length=1, max_length=5000)
    agent_type: str = Field(...)
    diagram_type: Optional[str] = Field(default="sequence")


class ChatResponse(BaseModel):
    """Modelo para respostas de chat"""
    response: str
    agent_type: str
    status: str
    sources: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None


class RAGKnowledgeRequest(BaseModel):
    """Modelo para adicionar conhecimento ao RAG"""
    url: Optional[str] = None
    text: Optional[str] = None
    source_id: Optional[str] = None


# Função para inicializar agentes
async def initialize_agents():
    """Inicializa os agentes disponíveis"""
    global agents_dict
    
    try:
        # MCP Agent
        if os.getenv("FIRECRAWL_API_KEY") and os.getenv("OPENAI_API_KEY"):
            try:
                agents_dict["mcp"] = MCPAgent()
                print("✅ MCP Agent inicializado")
            except Exception as e:
                print(f"❌ Erro ao inicializar MCP Agent: {e}")
        
        # Workflow Agent
        if os.getenv("FIRECRAWL_API_KEY") and os.getenv("OPENAI_API_KEY"):
            try:
                agents_dict["workflow"] = WorkflowAgent()
                print("✅ Workflow Agent inicializado")
            except Exception as e:
                print(f"❌ Erro ao inicializar Workflow Agent: {e}")
        
        # RAG Agent
        if os.getenv("PINECONE_API_KEY") and os.getenv("OPENAI_API_KEY"):
            try:
                rag_agent = RAGAgent()
                await rag_agent.initialize()
                agents_dict["rag"] = rag_agent
                print("✅ RAG Agent inicializado")
            except Exception as e:
                print(f"❌ Erro ao inicializar RAG Agent: {e}")
        
        # Externo Agent
        if os.getenv("OPENAI_API_KEY"):
            try:
                agents_dict["externo"] = ExternoAgent()
                print("✅ Agente Externo inicializado")
            except Exception as e:
                print(f"❌ Erro ao inicializar Agente Externo: {e}")
        
        # Tool Mermaid Agent
        if os.getenv("OPENAI_API_KEY"):
            try:
                agents_dict["mermaid"] = ToolMermaidAgent()
                print("✅ Tool Mermaid Agent inicializado")
            except Exception as e:
                print(f"❌ Erro ao inicializar Tool Mermaid Agent: {e}")
        
        # ClassificaImagem Agent (descomentar se o arquivo existir)
        # if os.getenv("OPENAI_API_KEY"):
        #     try:
        #         agents_dict["classifica_imagem"] = ClassificaImagemAgent()
        #         print("✅ ClassificaImagem Agent inicializado")
        #     except Exception as e:
        #         print(f"❌ Erro ao inicializar ClassificaImagem Agent: {e}")
        
    except Exception as e:
        print(f"❌ Erro geral na inicialização: {e}")


# Rotas
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Página inicial"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """Endpoint principal para chat"""
    
    agent = agents_dict.get(chat_request.agent_type)
    
    if not agent:
        return ChatResponse(
            response=f"❌ Agente {chat_request.agent_type} não está disponível.",
            agent_type=chat_request.agent_type,
            status="error"
        )
    
    try:
        if chat_request.agent_type == "mcp":
            response = await agent.process_message(chat_request.message)
            return ChatResponse(
                response=response,
                agent_type="mcp",
                status="success"
            )
        
        elif chat_request.agent_type == "workflow":
            response = await agent.process_query(chat_request.message)
            return ChatResponse(
                response=response,
                agent_type="workflow",
                status="success"
            )
        
        elif chat_request.agent_type == "rag":
            rag_response = await agent.query(chat_request.message)
            sources = []
            if rag_response.sources:
                sources = [
                    {
                        "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                        "score": doc.score,
                        "metadata": doc.metadata
                    }
                    for doc in rag_response.sources
                ]
            return ChatResponse(
                response=rag_response.answer,
                agent_type="rag",
                status="success",
                sources=sources,
                confidence=rag_response.confidence
            )
        
        elif chat_request.agent_type == "externo":
            response = await agent.process_message(chat_request.message)
            return ChatResponse(
                response=response,
                agent_type="externo",
                status="success"
            )
        
        elif chat_request.agent_type == "mermaid":
            diagram_type = getattr(chat_request, 'diagram_type', 'sequence')
            response = await agent.process_message(chat_request.message, diagram_type)
            return ChatResponse(
                response=response,
                agent_type="mermaid",
                status="success"
            )
        
        else:
            return ChatResponse(
                response="❌ Tipo de agente inválido",
                agent_type=chat_request.agent_type,
                status="error"
            )
            
    except Exception as e:
        print(f"Erro no chat: {e}")
        return ChatResponse(
            response=f"❌ Erro ao processar mensagem: {str(e)}",
            agent_type=chat_request.agent_type,
            status="error"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.5.0",
        "agents_available": list(agents_dict.keys()),
        "environment": os.getenv("ENVIRONMENT", "development")
    }


@app.get("/agents/info")
async def agents_info():
    """Informações sobre os agentes"""
    agents_info_list = [
        {
            "type": "mcp",
            "name": "Agente MCP Firecrawl",
            "description": "Scraping dinâmico com MCP",
            "available": "mcp" in agents_dict
        },
        {
            "type": "workflow",
            "name": "Agente Workflow",
            "description": "Pesquisa estruturada",
            "available": "workflow" in agents_dict
        },
        {
            "type": "rag",
            "name": "Agente RAG",
            "description": "RAG com Pinecone",
            "available": "rag" in agents_dict
        },
        {
            "type": "externo",
            "name": "Agente Externo",
            "description": "Integração com APIs externas",
            "available": "externo" in agents_dict
        },
        {
            "type": "mermaid",
            "name": "Tool Mermaid Agent",
            "description": "Geração de diagramas",
            "available": "mermaid" in agents_dict
        }
    ]
    
    return {"agents": agents_info_list}


# Endpoints RAG
@app.post("/rag/knowledge")
async def add_knowledge(request: RAGKnowledgeRequest):
    """Adicionar conhecimento ao RAG"""
    rag_agent = agents_dict.get("rag")
    
    if not rag_agent:
        raise HTTPException(status_code=503, detail="Agente RAG não disponível")
    
    try:
        if request.url:
            success = await rag_agent.add_knowledge_from_url(request.url)
            if success:
                return {"status": "success", "message": f"Conhecimento adicionado de: {request.url}"}
            else:
                return {"status": "error", "message": "Falha ao adicionar conhecimento"}
        
        elif request.text and request.source_id:
            success = await rag_agent.add_knowledge_from_text(request.text, request.source_id)
            if success:
                return {"status": "success", "message": f"Conhecimento adicionado: {request.source_id}"}
            else:
                return {"status": "error", "message": "Falha ao adicionar conhecimento"}
        
        else:
            raise HTTPException(status_code=400, detail="URL ou texto + source_id são obrigatórios")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")


if __name__ == "__main__":
    # Configuração para desenvolvimento
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )