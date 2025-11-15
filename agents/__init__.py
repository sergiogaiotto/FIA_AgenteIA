# agents/__init__.py - Comentado Linha a Linha

# Módulo agents - Agentes de IA para pesquisa e análise
# Comentário descritivo do propósito do módulo
# Define o domínio: agentes de IA especializados em pesquisa e análise

__version__ = "1.1.0"
# Versionamento semântico do módulo
# Formato: MAJOR.MINOR.PATCH
# 1.1.0 indica adição de nova funcionalidade (RAG Agent)
# Usado para controle de compatibilidade e releases

__author__ = "FIA.LabData - Prof Sergio Gaiotto"
# Metadado de autoria
# Identifica instituição (FIA.LabData) e responsável
# Importante para atribuição acadêmica e contato

from .mcp_agent import MCPAgent
# Importação relativa do agente MCP
# Ponto (.) indica importação do mesmo pacote
# MCPAgent: classe principal do agente que usa Model Context Protocol
# Conceito: separation of concerns - cada agente em módulo próprio

from .workflow_agent import WorkflowAgent
# Importação relativa do agente Workflow
# WorkflowAgent: classe para agente com fluxo estruturado
# Design pattern: cada tipo de agente implementa interface comum

from .rag_agent import RAGAgent
# Importação relativa do agente RAG
# RAGAgent: classe para Retrieval-Augmented Generation
# Nova funcionalidade: pesquisa semântica com Pinecone

from .externo_agent import ExternoAgent
# Importação relativa do agente Externo
# ExternoAgent: classe para integração com APIs externas (Flowise)
# Nova funcionalidade: processamento via APIs externas

from .tool_mermaid_agent import ToolMermaidAgent
# Importação relativa do agente Tool Mermaid
# ToolMermaidAgent: classe para geração de diagramas Mermaid
# Nova funcionalidade: criação de diagramas via MCP

__all__ = ["MCPAgent", "WorkflowAgent", "RAGAgent", "ExternoAgent", "ToolMermaidAgent"]
# Lista explícita de símbolos públicos do módulo
# Controla o que é importado com "from agents import *"
# Best practice: define API pública explicitamente
# Evita importação acidental de símbolos internos
# Documentação implícita das classes principais disponíveis
# ToolMermaidAgent na API pública