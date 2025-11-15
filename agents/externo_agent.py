# agents/externo_agent.py - Comentado Linha a Linha

# Agente Externo - Integração com API Flowise para processamento externo
# Comentário descritivo: define agente especializado em chamadas para APIs externas

import os
# Módulo padrão para interação com sistema operacional

import asyncio
# Biblioteca para programação assíncrona em Python

import aiohttp
# Cliente HTTP assíncrono para fazer requisições
# Melhor performance que requests em aplicações async

from typing import Dict, Any, Optional
# Type hints para melhor documentação e type safety

from pydantic import BaseModel, Field
# Framework para validação de dados e modelos tipados

# Carregamento de variáveis de ambiente
from dotenv import load_dotenv
# Carregamento de configuração de arquivo .env

# Carrega variáveis de ambiente
load_dotenv()


# ===============================
# MODELOS PYDANTIC
# ===============================

class FlowiseRequest(BaseModel):
    """Modelo para requisições ao Flowise"""
    question: str = Field(..., description="Pergunta a ser enviada para o Flowise")
    # Campo obrigatório: pergunta do usuário
    
    sessionId: Optional[str] = Field(default=None, description="ID da sessão (opcional)")
    # ID da sessão para manter contexto conversacional
    
    overrideConfig: Optional[Dict[str, Any]] = Field(default=None, description="Configurações customizadas")
    # Configurações adicionais para sobrescrever defaults


class FlowiseResponse(BaseModel):
    """Modelo para respostas do Flowise"""
    text: str = Field(..., description="Resposta gerada pelo Flowise")
    # Texto da resposta principal
    
    sourceDocuments: Optional[list] = Field(default=None, description="Documentos fonte utilizados")
    # Documentos que foram utilizados para gerar a resposta
    
    chatHistory: Optional[list] = Field(default=None, description="Histórico do chat")
    # Histórico da conversa para contexto


class ExternoAgentResponse(BaseModel):
    """Modelo para resposta estruturada do agente externo"""
    answer: str = Field(..., description="Resposta processada")
    # Resposta final formatada
    
    status: str = Field(..., description="Status da operação")
    # Status: success, error, warning
    
    sources: Optional[list] = Field(default=None, description="Fontes utilizadas")
    # Fontes que contribuíram para a resposta
    
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadados adicionais")
    # Informações extras sobre o processamento


# ===============================
# SERVIÇO FLOWISE
# ===============================

class FlowiseService:
    """Serviço para integração com API Flowise"""
    # Service class: encapsula operações da API Flowise
    # Abstração: isola complexidade da API externa
    
    def __init__(self, api_url: str = None):
        """Inicializa serviço Flowise"""
        # api_url: URL da API Flowise (pode ser customizada)

        # URL padrão fornecida pelo usuário
        self.api_url = api_url or os.getenv("API_EXTERNO_AGENT") # "https://gaiotto-flowiseai.hf.space/api/v1/prediction/126dd353-3c69-4304-9542-1263d07c711a"
        # URL completa da API Flowise com endpoint específico
        
        # Headers padrão para requisições
        self.headers = {
            "Content-Type": "application/json",
            # Content-Type para JSON payload
            "Accept": "application/json",
            # Accept header para especificar formato de resposta esperado
            "User-Agent": "FIA-AgentesIA/1.1.0"
            # User-Agent customizado para identificação
        }
        
        # Configurações de timeout
        self.timeout = aiohttp.ClientTimeout(total=300)
        # Timeout total de 30 segundos para evitar travamentos
        
        print(f"✅ Flowise Service inicializado com URL: {self.api_url}")
    
    async def query(self, payload: FlowiseRequest) -> FlowiseResponse:
        """Faz query para a API Flowise"""
        # Método principal para comunicação com Flowise
        # payload: dados estruturados para enviar
        
        try:
            # Converte payload Pydantic para dict
            data = payload.model_dump(exclude_none=True)
            # exclude_none: remove campos None do JSON
            # Evita enviar campos opcionais vazios
            
            print(f"🔍 Enviando query para Flowise: {payload.question[:100]}...")
            # Log da query (truncada para evitar logs longos)
            
            # Faz requisição assíncrona
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                # Context manager: garante cleanup automático da sessão
                
                async with session.post(
                    self.api_url,
                    json=data,
                    headers=self.headers
                ) as response:
                    # Context manager para response
                    # json=data: serializa automaticamente para JSON
                    
                    # Verifica status da resposta
                    if response.status == 200:
                        # Status 200: sucesso
                        
                        response_data = await response.json()
                        # Deserializa JSON response
                        
                        print(f"✅ Resposta recebida do Flowise (status: {response.status})")
                        
                        # Adapta resposta para formato esperado
                        return FlowiseResponse(
                            text=response_data.get("text", ""),
                            # Extrai texto principal (fallback string vazia)
                            sourceDocuments=response_data.get("sourceDocuments", []),
                            # Extrai documentos fonte (fallback lista vazia)
                            chatHistory=response_data.get("chatHistory", [])
                            # Extrai histórico (fallback lista vazia)
                        )
                    
                    else:
                        # Status diferente de 200: erro
                        error_text = await response.text()
                        # Lê corpo da resposta como texto para debugging
                        
                        print(f"❌ Erro na API Flowise (status: {response.status}): {error_text}")
                        
                        # Retorna resposta de erro estruturada
                        return FlowiseResponse(
                            text=f"Erro na API: Status {response.status}",
                            sourceDocuments=[],
                            chatHistory=[]
                        )
                        
        except asyncio.TimeoutError:
            # Timeout específico
            print("❌ Timeout na requisição para Flowise")
            return FlowiseResponse(
                text="❌ Timeout: A API Flowise demorou muito para responder. Tente novamente.",
                sourceDocuments=[],
                chatHistory=[]
            )
            
        except aiohttp.ClientError as e:
            # Erros de cliente HTTP
            print(f"❌ Erro de conexão com Flowise: {e}")
            return FlowiseResponse(
                text=f"❌ Erro de conexão: {str(e)}",
                sourceDocuments=[],
                chatHistory=[]
            )
            
        except Exception as e:
            # Erro genérico
            print(f"❌ Erro inesperado na integração Flowise: {e}")
            return FlowiseResponse(
                text=f"❌ Erro inesperado: {str(e)}",
                sourceDocuments=[],
                chatHistory=[]
            )
    
    async def health_check(self) -> bool:
        """Verifica se a API Flowise está respondendo"""
        # Método de diagnóstico para verificar conectividade
        
        try:
            # Faz query simples para testar conectividade
            test_payload = FlowiseRequest(question="test")
            # Query mínima para teste
            
            response = await self.query(test_payload)
            # Tenta fazer query de teste
            
            # Considera sucesso se não houve erro crítico
            return not response.text.startswith("❌")
            # Retorna True se resposta não começou com emoji de erro
            
        except Exception as e:
            print(f"❌ Health check falhou: {e}")
            return False
            # False indica que API não está acessível


# ===============================
# AGENTE EXTERNO PRINCIPAL
# ===============================

class ExternoAgent:
    """Agente especializado em integração com APIs externas (Flowise)"""
    # Classe principal que orquestra comunicação com Flowise
    # Abstração de alto nível para uso pela aplicação principal
    
    def __init__(self, api_url: str = None):
        """Inicializa agente externo"""
        
        # Inicializa serviço Flowise
        self.flowise_service = FlowiseService(api_url)
        # Dependency injection: serviço Flowise configurado
        
        # Histórico de mensagens para contexto
        self.message_history: list = []
        # Lista para manter contexto conversacional
        # Usado para gerar sessionId consistente
        
        # Configurações do agente
        self.session_id = f"fia-session-{os.getpid()}"
        # sessionId único baseado no PID do processo
        # Garante sessões únicas por instância da aplicação
        
        print(f"✅ Agente Externo inicializado com sessão: {self.session_id}")
    
    async def process_message(self, user_message: str) -> str:
        """
        Processa mensagem do usuário usando Flowise
        
        Args:
            user_message: Mensagem/consulta do usuário
            
        Returns:
            Resposta processada pelo Flowise
        """
        # Método principal: interface pública do agente
        # Async: permite operações não-bloqueantes
        
        if not user_message.strip():
            return "❌ Por favor, envie uma mensagem válida."
        # Validação de entrada: mensagem não pode estar vazia
        
        try:
            # Adiciona mensagem ao histórico
            self.message_history.append({
                "role": "user",
                "content": user_message,
                "timestamp": asyncio.get_event_loop().time()
                # Timestamp para tracking de sessão
            })
            
            # Mantém apenas últimas 10 mensagens para contexto
            if len(self.message_history) > 10:
                self.message_history = self.message_history[-10:]
            # Limita histórico para evitar crescimento excessivo
            
            # Prepara payload para Flowise
            flowise_request = FlowiseRequest(
                question=user_message,
                # Pergunta principal
                sessionId=self.session_id,
                # ID da sessão para contexto
                overrideConfig={
                    "returnSourceDocuments": True,
                    # Solicita retorno de documentos fonte
                    "returnChatHistory": True
                    # Solicita retorno do histórico
                }
            )
            
            # Faz query para Flowise
            flowise_response = await self.flowise_service.query(flowise_request)
            # Comunicação assíncrona com API externa
            
            # Adiciona resposta ao histórico
            self.message_history.append({
                "role": "assistant",
                "content": flowise_response.text,
                "timestamp": asyncio.get_event_loop().time(),
                "sources": len(flowise_response.sourceDocuments or [])
                # Contagem de fontes para metadados
            })
            
            # Formata resposta final
            formatted_response = self._format_response(flowise_response)
            
            return formatted_response
            
        except Exception as e:
            error_message = f"❌ Erro ao processar mensagem: {str(e)}"
            print(f"Erro Agente Externo: {e}")
            # Log de erro para debugging
            
            return error_message
            # Retorna erro amigável ao usuário
    
    def _format_response(self, flowise_response: FlowiseResponse) -> str:
        """Formata resposta do Flowise para exibição"""
        # Método privado: formatação interna
        # Melhora apresentação da resposta para o usuário
        
        # Resposta principal
        formatted = flowise_response.text
        
        # Adiciona informações sobre fontes se disponíveis
        if flowise_response.sourceDocuments:
            source_count = len(flowise_response.sourceDocuments)
            formatted += f"\n\n📚 **Baseado em {source_count} fonte(s)**"
            # Indica quantas fontes foram utilizadas
            
            # Mostra primeiras 2 fontes como exemplo
            for i, doc in enumerate(flowise_response.sourceDocuments[:2], 1):
                if isinstance(doc, dict) and 'pageContent' in doc:
                    # Verifica estrutura do documento
                    
                    content_preview = doc['pageContent'][:100]
                    # Preview do conteúdo (100 caracteres)
                    
                    formatted += f"\n{i}. {content_preview}..."
                    # Adiciona preview numerado
        
        # Adiciona informações sobre histórico se disponível
        if flowise_response.chatHistory:
            history_count = len(flowise_response.chatHistory)
            formatted += f"\n\n💬 **Contexto conversacional: {history_count} interações**"
            # Indica tamanho do contexto conversacional
        
        return formatted
    
    def reset_conversation(self):
        """Reseta histórico de conversa"""
        # Função utilitária para limpar contexto
        self.message_history = []
        # Limpa lista de mensagens
        
        # Gera novo session ID
        self.session_id = f"fia-session-{os.getpid()}-{asyncio.get_event_loop().time()}"
        # Novo sessionId com timestamp para unicidade
        
        print(f"🔄 Conversa resetada. Nova sessão: {self.session_id}")
    
    def get_conversation_history(self) -> list:
        """Retorna histórico da conversa"""
        # Método de acesso para histórico
        return self.message_history.copy()
        # .copy(): retorna cópia para evitar modificação externa
    
    async def check_service_availability(self) -> Dict[str, Any]:
        """
        Verifica disponibilidade do serviço Flowise
        
        Returns:
            Dicionário com status do serviço
        """
        # Função de diagnóstico para verificar configuração
        
        try:
            # Testa conectividade
            is_available = await self.flowise_service.health_check()
            
            return {
                "status": "available" if is_available else "unavailable",
                # Status baseado no health check
                "service": "Flowise API",
                # Nome do serviço
                "endpoint": self.flowise_service.api_url,
                # URL sendo utilizada
                "session_id": self.session_id,
                # ID da sessão atual
                "message_count": len(self.message_history)
                # Quantidade de mensagens na sessão
            }
            
        except Exception as e:
            # Falha na verificação
            return {
                "status": "error",
                "service": "Flowise API",
                "error": str(e),
                "endpoint": self.flowise_service.api_url,
                "session_id": self.session_id,
                "message_count": len(self.message_history)
            }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o agente"""
        # Método de introspecção: informações sobre capacidades
        
        return {
            "name": "Agente Externo (Flowise)",
            "version": "1.0.0",
            "type": "external_api",
            "capabilities": [
                "Integração com Flowise API",
                "Processamento de linguagem natural",
                "Manutenção de contexto conversacional",
                "Análise de documentos fonte",
                "Formatação automática de respostas"
            ],
            "configuration": {
                "api_url": self.flowise_service.api_url,
                "session_management": True,
                "source_tracking": True,
                "context_limit": 10
                # Limite de mensagens no contexto
            }
        }
        # Metadados estruturados sobre o agente