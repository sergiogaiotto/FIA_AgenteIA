"""
Agent Factory - Criação e gerenciamento centralizado de agentes
"""

import asyncio
from typing import Dict, Optional, Any, List
from abc import ABC, abstractmethod
import logging

from config import settings
from agents.mcp_agent import MCPAgent
from agents.workflow_agent import WorkflowAgent
from agents.rag_agent import RAGAgent
from agents.externo_agent import ExternoAgent
from agents.tool_mermaid_agent import ToolMermaidAgent
from agents.classifica_imagem_agent import ClassificaImagemAgent

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all agents"""
    
    @abstractmethod
    async def process_message(self, message: str, **kwargs) -> Dict[str, Any]:
        """Process a message and return result"""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Cleanup resources"""
        pass


class AgentFactory:
    """Factory for creating and managing agents"""
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._initialization_errors: Dict[str, str] = {}
        
    async def initialize_agents(self):
        """Initialize all available agents"""
        initialization_tasks = []
        
        # MCP Agent
        if settings.firecrawl_api_key and settings.openai_api_key:
            initialization_tasks.append(
                self._initialize_agent("mcp", MCPAgent)
            )
        else:
            self._initialization_errors["mcp"] = "Missing API keys"
        
        # Workflow Agent
        if settings.firecrawl_api_key and settings.openai_api_key:
            initialization_tasks.append(
                self._initialize_agent("workflow", WorkflowAgent)
            )
        else:
            self._initialization_errors["workflow"] = "Missing API keys"
        
        # RAG Agent
        if settings.pinecone_api_key and settings.openai_api_key:
            initialization_tasks.append(
                self._initialize_agent("rag", RAGAgent)
            )
        else:
            self._initialization_errors["rag"] = "Missing Pinecone API key"
        
        # Externo Agent
        if settings.openai_api_key:
            initialization_tasks.append(
                self._initialize_agent("externo", ExternoAgent)
            )
        
        # Tool Mermaid Agent
        if settings.openai_api_key:
            initialization_tasks.append(
                self._initialize_agent("mermaid", ToolMermaidAgent)
            )
        
        # ClassificaImagem Agent
        if settings.openai_api_key:
            initialization_tasks.append(
                self._initialize_agent("classifica_imagem", ClassificaImagemAgent)
            )
        
        # Run all initializations concurrently
        if initialization_tasks:
            await asyncio.gather(*initialization_tasks, return_exceptions=True)
    
    async def _initialize_agent(self, agent_type: str, agent_class):
        """Initialize a single agent"""
        try:
            agent = agent_class()
            if hasattr(agent, 'initialize'):
                await agent.initialize()
            self._agents[agent_type] = agent
            logger.info(f"✅ {agent_type} agent initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize: {str(e)}"
            self._initialization_errors[agent_type] = error_msg
            logger.error(f"❌ {agent_type} agent: {error_msg}")
    
    def get_agent(self, agent_type: str) -> Optional[BaseAgent]:
        """Get an agent by type"""
        return self._agents.get(agent_type)
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agent types"""
        return list(self._agents.keys())
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status = {}
        for agent_type in ["mcp", "workflow", "rag", "externo", "mermaid", "classifica_imagem"]:
            if agent_type in self._agents:
                status[agent_type] = "available"
            elif agent_type in self._initialization_errors:
                status[agent_type] = f"error: {self._initialization_errors[agent_type]}"
            else:
                status[agent_type] = "not configured"
        return status
    
    def get_agents_info(self) -> List[Dict[str, Any]]:
        """Get information about all agents"""
        agents_info = [
            {
                "type": "mcp",
                "name": "Agente MCP Firecrawl",
                "description": "Scraping dinâmico com Model Context Protocol",
                "features": ["Scraping em tempo real", "Integração MCP", "Análise conversacional"],
                "available": "mcp" in self._agents
            },
            {
                "type": "workflow",
                "name": "Agente Workflow",
                "description": "Pesquisa estruturada e análise comparativa",
                "features": ["Workflow estruturado", "Análise comparativa", "Recomendações"],
                "available": "workflow" in self._agents
            },
            {
                "type": "rag",
                "name": "Agente RAG",
                "description": "Retrieval-Augmented Generation com Pinecone",
                "features": ["Pesquisa semântica", "Base de conhecimento", "Citação de fontes"],
                "available": "rag" in self._agents
            },
            {
                "type": "externo",
                "name": "Agente Externo",
                "description": "Integração com APIs externas (Flowise)",
                "features": ["APIs externas", "Contexto conversacional", "Formatação automática"],
                "available": "externo" in self._agents
            },
            {
                "type": "mermaid",
                "name": "Tool Mermaid Agent",
                "description": "Geração de diagramas Mermaid",
                "features": ["Diagramas de sequência", "Fluxogramas", "Gráficos Gantt"],
                "available": "mermaid" in self._agents
            },
            {
                "type": "classifica_imagem",
                "name": "ClassificaImagem Agent",
                "description": "Análise visual com GPT-4 Vision",
                "features": ["Detecção de objetos", "Análise de cores", "Insights de marketing"],
                "available": "classifica_imagem" in self._agents
            }
        ]
        return agents_info
    
    async def cleanup(self):
        """Cleanup all agents"""
        cleanup_tasks = []
        for agent in self._agents.values():
            if hasattr(agent, 'cleanup'):
                cleanup_tasks.append(agent.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self._agents.clear()
        logger.info("All agents cleaned up")