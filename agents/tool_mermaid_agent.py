# agents/tool_mermaid_agent.py - Comentado Linha a Linha

# Agente Tool Mermaid - Geração de diagramas usando Mermaid via MCP
# Comentário descritivo: agente especializado em criar diagramas de sequência e outros tipos

import os
# Módulo padrão para interação com sistema operacional

import asyncio
# Biblioteca para programação assíncrona

import json
# Módulo para manipulação de dados JSON

from typing import List, Dict, Any, Optional
# Type hints para melhor documentação e type safety

# Imports do MCP (Model Context Protocol)
from mcp import ClientSession, StdioServerParameters
# ClientSession: classe para gerenciar sessão MCP
# StdioServerParameters: configuração para comunicação via stdio

from mcp.client.stdio import stdio_client
# Cliente MCP para comunicação via standard input/output

# LangGraph para agentes ReAct
# LangGraph: framework para criar fluxos de agentes complexos

# Modelo OpenAI
from langchain_openai import ChatOpenAI
# Integração LangChain com API OpenAI

from langchain_core.messages import HumanMessage, SystemMessage
# Tipos de mensagem padronizados do LangChain

# Pydantic para modelos de dados
from pydantic import BaseModel, Field
# BaseModel: validação e serialização
# Field: metadados para campos

# Carregamento de variáveis de ambiente
from dotenv import load_dotenv
# Biblioteca para carregar variáveis de arquivo .env

# Carrega variáveis de ambiente
load_dotenv()


# ===============================
# MODELOS PYDANTIC
# ===============================

class MermaidRequest(BaseModel):
    """Modelo para requisições de geração de diagrama"""
    prompt: str = Field(..., description="Descrição do diagrama em linguagem natural")
    # Campo obrigatório: descrição do que o usuário quer no diagrama
    
    diagram_type: str = Field(default="sequence", description="Tipo de diagrama Mermaid")
    # Tipo padrão: sequence diagram, mas pode ser flowchart, gantt, etc.
    
    style: Optional[str] = Field(default=None, description="Estilo personalizado do diagrama")
    # Estilo opcional para customização visual


class MermaidResponse(BaseModel):
    """Modelo para resposta do diagrama gerado"""
    mermaid_code: str = Field(..., description="Código Mermaid gerado")
    # Código Mermaid que será renderizado
    
    explanation: str = Field(..., description="Explicação do diagrama")
    # Explicação em linguagem natural do que o diagrama representa
    
    diagram_type: str = Field(..., description="Tipo de diagrama gerado")
    # Tipo de diagrama que foi criado
    
    suggestions: List[str] = Field(default_factory=list, description="Sugestões de melhorias")
    # Lista de sugestões para melhorar o diagrama


# ===============================
# PROMPTS ORGANIZADOS
# ===============================

class MermaidPrompts:
    """Container para prompts organizados por categoria"""
    
    # System prompt para geração de diagramas
    MERMAID_GENERATION_SYSTEM = """Você é um especialista em criação de diagramas Mermaid. 
    Sua função é converter descrições em linguagem natural em diagramas Mermaid bem estruturados.
    
    Capabilities:
    - Sequence Diagrams: para mostrar interações entre entidades
    - Flowcharts: para mostrar fluxos de processo
    - Class Diagrams: para mostrar estruturas OOP
    - State Diagrams: para mostrar máquinas de estado
    - Gantt Charts: para mostrar cronogramas
    - ER Diagrams: para mostrar modelos de dados
    
    Sempre:
    1. Gere código Mermaid válido e bem formatado
    2. Use nomes descritivos para elementos
    3. Inclua comentários quando necessário
    4. Mantenha o diagrama claro e legível
    5. Sugira melhorias quando apropriado
    
    Output Format:
    - Código Mermaid completo e válido
    - Explicação clara do diagrama
    - Sugestões de melhorias se aplicável"""

    @staticmethod
    def generate_diagram_prompt(prompt: str, diagram_type: str) -> str:
        """Template para geração de diagramas"""
        return f"""User Request: {prompt}
        
Requested Diagram Type: {diagram_type}

Please generate a Mermaid diagram based on this description. Follow these guidelines:

1. Create valid Mermaid syntax for {diagram_type}
2. Use clear, descriptive labels
3. Ensure proper flow and structure
4. Add styling when appropriate
5. Make it professional and easy to understand

Provide:
1. Complete Mermaid code (wrapped in ```mermaid ... ```)
2. Brief explanation of the diagram
3. Any suggestions for improvements or alternatives

Focus on clarity and accuracy."""


# ===============================
# AGENTE TOOL MERMAID PRINCIPAL
# ===============================

class ToolMermaidAgent:
    """Agente especializado em geração de diagramas Mermaid via MCP"""
    # Classe principal que orquestra geração de diagramas
    
    def __init__(self):
        """Inicializa agente Mermaid com configurações necessárias"""
        
        # Validação de chaves de API
        self.openai_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_key:
            raise ValueError("OPENAI_API_KEY não encontrada nas variáveis de ambiente")
        
        # Configuração do modelo LLM
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            # Modelo otimizado para tarefas de geração de código
            temperature=0.1,
            # Baixa temperatura para consistência em geração de código
            openai_api_key=self.openai_key
        )
        
        # Configuração do servidor MCP (pode usar ferramentas MCP se disponíveis)
        firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
        if firecrawl_key:
            self.server_params = StdioServerParameters(
                command="npx",
                env={"FIRECRAWL_API_KEY": firecrawl_key},
                args=["firecrawl-mcp"]
            )
            self.mcp_available = True
        else:
            self.server_params = None
            self.mcp_available = False
        
        # Prompts organizados
        self.prompts = MermaidPrompts()
        
        # Histórico de diagramas gerados
        self.diagram_history: List[Dict[str, Any]] = []
        
        print("✅ Tool Mermaid Agent inicializado")
    
    async def generate_diagram(self, request: MermaidRequest) -> MermaidResponse:
        """
        Gera diagrama Mermaid baseado na requisição
        
        Args:
            request: Requisição com descrição do diagrama
            
        Returns:
            Resposta com código Mermaid e explicação
        """
        try:
            print(f"🎨 Gerando diagrama {request.diagram_type}: {request.prompt[:100]}...")
            
            # Prepara mensagens para o LLM
            messages = [
                SystemMessage(content=self.prompts.MERMAID_GENERATION_SYSTEM),
                HumanMessage(content=self.prompts.generate_diagram_prompt(
                    request.prompt, 
                    request.diagram_type
                ))
            ]
            
            # Gera resposta usando LLM
            response = self.model.invoke(messages)
            response_content = response.content
            
            # Extrai código Mermaid da resposta
            mermaid_code = self._extract_mermaid_code(response_content)
            
            # Gera explicação
            explanation = self._extract_explanation(response_content)
            
            # Gera sugestões
            suggestions = self._extract_suggestions(response_content)
            
            # Cria resposta estruturada
            mermaid_response = MermaidResponse(
                mermaid_code=mermaid_code,
                explanation=explanation,
                diagram_type=request.diagram_type,
                suggestions=suggestions
            )
            
            # Adiciona ao histórico
            self.diagram_history.append({
                "timestamp": asyncio.get_event_loop().time(),
                "request": request.model_dump(),
                "response": mermaid_response.model_dump()
            })
            
            print(f"✅ Diagrama {request.diagram_type} gerado com sucesso")
            return mermaid_response
            
        except Exception as e:
            print(f"❌ Erro ao gerar diagrama: {e}")
            return MermaidResponse(
                mermaid_code="graph TD\n    A[Erro] --> B[Falha na geração]",
                explanation=f"Erro ao gerar diagrama: {str(e)}",
                diagram_type=request.diagram_type,
                suggestions=["Tente reformular a descrição", "Verifique se o tipo de diagrama está correto"]
            )
    
    def _extract_mermaid_code(self, response: str) -> str:
        """Extrai código Mermaid da resposta do LLM"""
        # Procura por blocos de código Mermaid
        import re
        
        # Pattern para ```mermaid ... ```
        mermaid_pattern = r'```mermaid\s*(.*?)\s*```'
        match = re.search(mermaid_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # Pattern alternativo para ``` ... ``` após mencionar mermaid
        code_pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            # Retorna o primeiro bloco de código encontrado
            return matches[0].strip()
        
        # Fallback: tenta extrair linhas que parecem código Mermaid
        lines = response.split('\n')
        mermaid_lines = []
        in_diagram = False
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['graph', 'sequenceDiagram', 'classDiagram', 'stateDiagram', 'gantt']):
                in_diagram = True
            
            if in_diagram and line:
                mermaid_lines.append(line)
            elif in_diagram and not line:
                break
        
        if mermaid_lines:
            return '\n'.join(mermaid_lines)
        
        # Último fallback: exemplo básico baseado no tipo
        return self._get_fallback_diagram()
    
    def _extract_explanation(self, response: str) -> str:
        """Extrai explicação da resposta do LLM"""
        # Procura por seções de explicação
        lines = response.split('\n')
        explanation_lines = []
        capturing = False
        
        for line in lines:
            line = line.strip()
            
            # Identifica início de explicação
            if any(keyword in line.lower() for keyword in ['explanation', 'explicação', 'descrição', 'este diagrama']):
                capturing = True
                if line.lower().startswith(('explanation', 'explicação')):
                    continue
            
            # Captura linhas da explicação
            if capturing and line and not line.startswith('```'):
                explanation_lines.append(line)
            elif capturing and not line:
                break
        
        explanation = ' '.join(explanation_lines)
        
        if explanation:
            return explanation
        
        # Fallback: usa as primeiras linhas não-código como explicação
        non_code_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('```') and not any(keyword in line for keyword in ['graph', 'sequenceDiagram']):
                non_code_lines.append(line)
                if len(non_code_lines) >= 3:
                    break
        
        return ' '.join(non_code_lines) if non_code_lines else "Diagrama gerado conforme solicitado."
    
    def _extract_suggestions(self, response: str) -> List[str]:
        """Extrai sugestões da resposta do LLM"""
        lines = response.split('\n')
        suggestions = []
        capturing = False
        
        for line in lines:
            line = line.strip()
            
            # Identifica seção de sugestões
            if any(keyword in line.lower() for keyword in ['suggestions', 'sugestões', 'melhorias', 'improvements']):
                capturing = True
                continue
            
            # Captura sugestões (linhas com bullets ou números)
            if capturing and line:
                if line.startswith(('-', '*', '•')) or line[0].isdigit():
                    # Remove bullets e números
                    suggestion = line.lstrip('-*•0123456789. ').strip()
                    if suggestion:
                        suggestions.append(suggestion)
                elif not line:
                    break
        
        # Fallback: sugestões padrão
        if not suggestions:
            suggestions = [
                "Considere adicionar mais detalhes se necessário",
                "Verifique se todos os elementos estão claramente rotulados",
                "Teste a renderização do diagrama"
            ]
        
        return suggestions
    
    def _get_fallback_diagram(self) -> str:
        """Retorna diagrama de fallback em caso de erro"""
        return """graph TD
    A[Início] --> B[Processo]
    B --> C[Decisão]
    C -->|Sim| D[Ação 1]
    C -->|Não| E[Ação 2]
    D --> F[Fim]
    E --> F"""
    
    async def process_message(self, user_message: str, diagram_type: str = "sequence") -> str:
        """
        Processa mensagem do usuário e gera diagrama
        
        Args:
            user_message: Descrição do diagrama desejado
            diagram_type: Tipo de diagrama (sequence, flowchart, etc.)
            
        Returns:
            Resposta formatada com código Mermaid e explicação
        """
        try:
            # Cria requisição
            request = MermaidRequest(
                prompt=user_message,
                diagram_type=diagram_type
            )
            
            # Gera diagrama
            response = await self.generate_diagram(request)
            
            # Formata resposta para exibição
            formatted_response = self._format_response(response)
            
            return formatted_response
            
        except Exception as e:
            return f"❌ Erro ao processar solicitação: {str(e)}"
    
    def _format_response(self, response: MermaidResponse) -> str:
        """Formata resposta para exibição no chat"""
        formatted = f"""🎨 **Diagrama {response.diagram_type.title()} Gerado**

📝 **Explicação:**
{response.explanation}

📊 **Código Mermaid:**
```mermaid
{response.mermaid_code}
```

💡 **Sugestões:**"""
        
        for i, suggestion in enumerate(response.suggestions, 1):
            formatted += f"\n{i}. {suggestion}"
        
        return formatted
    
    def get_supported_diagrams(self) -> List[Dict[str, str]]:
        """Retorna lista de tipos de diagrama suportados"""
        return [
            {"type": "sequence", "name": "Diagrama de Sequência", "description": "Interações entre entidades ao longo do tempo"},
            {"type": "flowchart", "name": "Fluxograma", "description": "Fluxo de processos e decisões"},
            {"type": "classDiagram", "name": "Diagrama de Classes", "description": "Estruturas orientadas a objetos"},
            {"type": "stateDiagram", "name": "Diagrama de Estados", "description": "Máquinas de estado e transições"},
            {"type": "gantt", "name": "Gráfico de Gantt", "description": "Cronogramas e planejamento"},
            {"type": "erDiagram", "name": "Diagrama ER", "description": "Modelos de dados e relacionamentos"},
            {"type": "journey", "name": "Jornada do Usuário", "description": "Experiência do usuário passo a passo"}
        ]
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o agente"""
        return {
            "name": "Tool Mermaid Agent",
            "version": "1.0.0",
            "type": "diagram_generation",
            "capabilities": [
                "Geração de diagramas Mermaid",
                "Múltiplos tipos de diagrama",
                "Explicações detalhadas",
                "Sugestões de melhorias",
                "Histórico de diagramas"
            ],
            "supported_diagrams": [d["type"] for d in self.get_supported_diagrams()],
            "mcp_integration": self.mcp_available,
            "diagrams_generated": len(self.diagram_history)
        }
    
    def get_diagram_history(self) -> List[Dict[str, Any]]:
        """Retorna histórico de diagramas gerados"""
        return self.diagram_history.copy()
    
    def reset_conversation(self):
        """Reseta histórico de conversa"""
        self.diagram_history = []
        print("🔄 Histórico de diagramas resetado")