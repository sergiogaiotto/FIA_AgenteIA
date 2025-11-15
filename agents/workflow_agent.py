# agents/workflow_agent.py - Comentado Linha a Linha

# Agente Workflow - Pesquisa estruturada e análise comparativa
# Comentário descritivo: define especialização em workflows estruturados

import os
# Módulo padrão para interação com sistema operacional

import json
# Módulo para manipulação de dados JSON
# Usado para serialização/deserialização de estruturas complexas

from typing import Dict, Any, List, Optional
# Type hints para melhor documentação e type safety

# Imports para workflow estruturado
from langgraph.graph import StateGraph, END
# StateGraph: grafo de estados para workflows complexos
# END: nó terminal que finaliza execução do workflow
# Conceito: State Machine para fluxos determinísticos

from langchain_openai import ChatOpenAI
# Integração com modelos OpenAI via LangChain

from langchain_core.messages import HumanMessage, SystemMessage
# Tipos de mensagem padronizados do LangChain
# HumanMessage: entrada do usuário
# SystemMessage: instruções do sistema

# Imports dos modelos Pydantic
from pydantic import BaseModel
# Framework para validação de dados e modelos tipados
# Benefícios: validação automática, serialização, type safety

# Carregamento de variáveis
from dotenv import load_dotenv
# Carregamento de configuração de arquivo .env

# Biblioteca Firecrawl
from firecrawl import FirecrawlApp
# SDK oficial para API Firecrawl
# Abstração de alto nível para web scraping

# Carrega variáveis de ambiente
load_dotenv()
# Execução do carregamento de configurações


# ===============================
# MODELOS PYDANTIC
# ===============================
# Seção organizacional: agrupa definições de modelos de dados

class CompanyAnalysis(BaseModel):
    """Modelo para análise estruturada de empresa/produto"""
    # Pydantic model para dados estruturados de análise
    # BaseModel: classe base que fornece validação e serialização
    
    pricing_model: str
    # Campo obrigatório: modelo de precificação da empresa
    # str: tipo string obrigatório
    
    is_open_source: Optional[bool] = None
    # Campo opcional: indica se é open source
    # Optional[bool]: pode ser True, False ou None
    # Default None: valor padrão quando não especificado
    
    tech_stack: List[str] = []
    # Lista de tecnologias utilizadas
    # List[str]: lista de strings
    # Default []: lista vazia como padrão
    
    description: str = ""
    # Descrição textual da empresa/produto
    # Default "": string vazia como padrão
    
    api_available: Optional[bool] = None
    # Indica disponibilidade de API
    # Pattern repetido: Optional para dados incertos
    
    language_support: List[str] = []
    # Linguagens de programação suportadas
    # Lista vazia por padrão
    
    integration_capabilities: List[str] = []
    # Capacidades de integração com outras ferramentas
    # Pattern consistente: listas vazias por padrão


class CompanyInfo(BaseModel):
    """Modelo completo para informações de empresa"""
    # Modelo mais abrangente que inclui CompanyAnalysis
    
    name: str
    # Nome da empresa (campo obrigatório)
    
    description: str
    # Descrição da empresa (campo obrigatório)
    
    website: str
    # URL do website oficial (campo obrigatório)
    
    pricing_model: Optional[str] = None
    # Modelo de preços (opcional, pode não estar disponível)
    
    is_open_source: Optional[bool] = None
    # Indica licenciamento open source
    
    tech_stack: List[str] = []
    # Stack tecnológico utilizado
    
    competitors: List[str] = []
    # Lista de concorrentes identificados
    
    api_available: Optional[bool] = None
    # Disponibilidade de API pública
    
    language_support: List[str] = []
    # Suporte a linguagens de programação
    
    integration_capabilities: List[str] = []
    # Integrações suportadas
    
    developer_experience_rating: Optional[str] = None
    # Avaliação da experiência do desenvolvedor
    # String para permitir ratings qualitativos


class ResearchState(BaseModel):
    """Modelo para estado global do workflow"""
    # State object para workflow LangGraph
    # Mantém estado através das etapas do workflow
    
    query: str
    # Consulta original do usuário (imutável)
    
    extracted_tools: List[str] = []
    # Ferramentas extraídas de artigos
    # Resultado da primeira etapa do workflow
    
    companies: List[CompanyInfo] = []
    # Informações coletadas das empresas
    # Resultado da etapa de pesquisa
    
    search_results: List[Dict[str, Any]] = []
    # Resultados brutos de busca
    # Dict[str, Any]: estrutura flexível para dados de APIs
    
    analysis: Optional[str] = None
    # Análise final e recomendações
    # Resultado da última etapa do workflow


# ===============================
# PROMPTS ORGANIZADOS
# ===============================
# Seção organizacional: centraliza prompts em classe

class DeveloperToolsPrompts:
    """Container para prompts organizados por categoria"""
    # Design pattern: namespace para prompts
    # Vantagem: organização, reutilização, manutenção
    
    # Tool extraction prompts
    TOOL_EXTRACTION_SYSTEM = """Você é um pesquisador de preços, promoções, ofertas e valores. Extraia nomes específicos de ferramentas, bibliotecas, plataformas ou serviços de artigos.
Concentre-se em produtos/ferramentas/soluções/serviços reais que consumidores demonstrem interesse e podem usar."""
    # System prompt para extração de ferramentas
    # Define persona: pesquisador especializado
    # Foco: produtos comercializáveis para consumidores

    @staticmethod
    def tool_extraction_user(query: str, content: str) -> str:
        # Static method: não precisa de instância da classe
        # Template function: gera prompt personalizado
        return f"""Query: {query}
Conteúdo do Artigo: {content}

Extraia uma lista de nomes de produtos/ferramentas/soluções/serviços específicos mencionados neste conteúdo que sejam relevantes para "{query}".

Rules:
- Incluir apenas nomes de produtos reais, sem termos genéricos
- Foco em ferramentas/soluções/serviços que os consumidores podem comprar, obter, assinar, usar e consumir diretamente
- Incluir opções comerciais e de código aberto
- Limitar às 5 resultados mais relevantes
- Retornar apenas os nomes dos produtos/ferramentas/soluções/serviços, um por linha, sem descrições

Formato de exemplo:
Amazon
MercadoLivre
Picpay
Nubank
Microsoft
"""
        # Template detalhado com:
        # 1. Context injection (query + content)
        # 2. Instruções específicas
        # 3. Constraints (5 resultados, formato específico)
        # 4. Examples para few-shot learning

    # Company analysis prompts
    TOOL_ANALYSIS_SYSTEM = """Você está analisando preços, promoções, ofertas e valores de produtos/ferramentas/soluções/serviços com base na categoria informada pelo usuário.
Concentre-se em extrair informações relevantes para consumidores de produtos/ferramentas/soluções/serviços.
Preste atenção especial nas condições, descontos, modelo comercial, pré-requisitos, tecnologia, APIs, SDKs e modos de utilização."""
    # System prompt para análise de empresas
    # Foca em aspectos comerciais e técnicos relevantes

    @staticmethod
    def tool_analysis_user(company_name: str, content: str) -> str:
        # Template para análise individual de empresa
        return f"""Empresa/Ferramenta: {company_name}
Conteúdo do Website: {content[:2500]}

Analise este conteúdo da perspectiva de um consumidor e forneça:
- pricing_model: "Gratuito", "Freemium", "Pago", "Empresarial", "Assinatura" ou "Desconhecido"
- is_open_source: verdadeiro se for de código aberto, falso se for proprietário, nulo se não estiver claro
- tech_stack: tecnologia adotada para produtos/ferramentas/soluções/serviços oferecido
- description: breve descrição de uma frase com foco no que esta produtos/ferramentas/soluções/serviços entrega para o consumidor
- api_available: verdadeiro se API REST, GraphQL, SDK ou acesso programático forem mencionados
- language_support: Lista de linguagens de programação explicitamente suportadas
- integration_capabilities: Lista de ferramentas/plataformas com as quais se integra
"""
        # Template estruturado que:
        # 1. Limita conteúdo ([:2500]) para evitar overflow
        # 2. Define campos específicos do modelo Pydantic
        # 3. Fornece valores válidos para campos categóricos
        # 4. Especifica critérios de avaliação

    # Recommendations prompts
    RECOMMENDATIONS_SYSTEM = """Você é um pesquisador sênior que fornece recomendações técnicas rápidas e concisas.
Mantenha as respostas breves e práticas - no máximo 3 a 4 frases no total."""
    # System prompt para geração de recomendações
    # Constraint de brevidade: 3-4 frases
    # Persona: pesquisador sênior (autoridade)

    @staticmethod
    def recommendations_user(query: str, company_data: str) -> str:
        # Template para geração de recomendações finais
        return f"""Consumer Query: {query}
Ferramentas/Tecnologias Analisadas: {company_data}

Forneça uma breve recomendação (máximo de 3 a 4 frases) abrangendo:
- Qual ferramenta é a melhor e por quê
- Principais considerações sobre custo/preço
- Principal vantagem técnica
- A melhor oferta, preços e condições

Não são necessárias longas explicações."""
        # Template focado em:
        # 1. Decisão clara (melhor ferramenta)
        # 2. Aspectos financeiros (custo/preço)
        # 3. Aspectos técnicos (vantagens)
        # 4. Aspectos comerciais (ofertas)


# ===============================
# SERVIÇO FIRECRAWL
# ===============================
# Seção organizacional: abstração para serviços externos

class FirecrawlService:
    """Serviço para integração com Firecrawl API"""
    # Service class: encapsula integração com API externa
    # Design pattern: Facade para simplificar uso de API complexa
    
    def __init__(self):
        api_key = os.getenv("FIRECRAWL_API_KEY")
        # Obtém chave da API das variáveis de ambiente
        
        if not api_key:
            raise ValueError("FIRECRAWL_API_KEY não encontrada")
        # Fail-fast: falha imediatamente se configuração inválida
        
        self.app = FirecrawlApp(api_key=api_key)
        # Inicializa cliente Firecrawl com autenticação

    def search_companies(self, query: str, num_results: int = 5):
        """Busca empresas/produtos usando Firecrawl"""
        # Método para busca de empresas
        # Default num_results=5: valor padrão otimizado
        
        try:
            result = self.app.search(
                query=f"{query} preços, ofertas e valores",
                # Augmented query: adiciona termos comerciais
                # Melhora relevância dos resultados
                limit=num_results
                # Limita quantidade de resultados
            )
            return result
            # Retorna resultado bruto da API
            
        except Exception as e:
            print(f"Erro na busca: {e}")
            # Log do erro para debugging
            return []
            # Retorna lista vazia em caso de erro
            # Graceful degradation: falha não quebra workflow

    def scrape_company_pages(self, url: str):
        """Faz scraping de páginas específicas"""
        # Método para scraping de URL específica
        
        try:
            result = self.app.scrape_url(
                url,
                formats=["markdown"]
                # Formato markdown: estruturado, fácil de processar
                # Preserva hierarquia de conteúdo
            )
            return result
            # Retorna conteúdo estruturado
            
        except Exception as e:
            print(f"Erro no scraping: {e}")
            # Log de erro para monitoramento
            return None
            # Retorna None para indicar falha
            # Permite verificação simples: if scraped:


# ===============================
# AGENTE WORKFLOW PRINCIPAL
# ===============================
# Seção principal: implementação do agente workflow

class WorkflowAgent:
    """Agente de pesquisa estruturada usando workflows LangGraph"""
    # Classe principal que orquestra todo o processo

    def __init__(self):
        """Inicializa agente com dependências necessárias"""
        
        # Validação de chaves de API
        if not os.getenv("FIRECRAWL_API_KEY"):
            raise ValueError("FIRECRAWL_API_KEY não encontrada")
        # Validação crítica: sem API key, agente não funciona
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY não encontrada")
        # Dupla validação: ambas APIs necessárias
        
        # Inicializa serviços
        self.firecrawl = FirecrawlService()
        # Dependency injection: serviço Firecrawl
        
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)
        # LLM com configuração otimizada:
        # - gpt-4.1-mini: modelo eficiente
        # - temperature=0.1: quase determinístico, mas com pequena variação
        
        self.prompts = DeveloperToolsPrompts()
        # Namespace de prompts organizados
        
        # Constrói workflow
        self.workflow = self._build_workflow()
        # Inicialização do grafo de estados

    def _build_workflow(self):
        """Constrói workflow como máquina de estados"""
        # Método privado para construção do workflow
        # _ prefix: convenção Python para métodos internos
        
        graph = StateGraph(ResearchState)
        # Cria grafo com ResearchState como tipo do estado
        # StateGraph: framework LangGraph para workflows
        
        # Adiciona nós do workflow
        graph.add_node("extract_tools", self._extract_tools_step)
        # Nó 1: extração de ferramentas de artigos
        
        graph.add_node("research", self._research_step)
        # Nó 2: pesquisa detalhada de cada ferramenta
        
        graph.add_node("analyze", self._analyze_step)
        # Nó 3: análise e geração de recomendações
        
        # Define transições
        graph.set_entry_point("extract_tools")
        # Ponto de entrada: sempre começa por extract_tools
        
        graph.add_edge("extract_tools", "research")
        # Transição linear: extract_tools → research
        
        graph.add_edge("research", "analyze")
        # Transição linear: research → analyze
        
        graph.add_edge("analyze", END)
        # Fim do workflow: analyze → END
        # END: constante que termina execução
        
        return graph.compile()
        # Compila grafo em workflow executável

    def _extract_tools_step(self, state: ResearchState) -> Dict[str, Any]:
        """Primeiro passo: extrai ferramentas de artigos"""
        # Step function: recebe state, retorna updates
        # Pattern LangGraph: funções puras que modificam estado
        
        print(f"🔍 Buscando artigos sobre: {state.query}")
        # UX feedback: informa progresso ao usuário
        # Emoji visual: melhora experiência

        # Busca artigos relevantes
        article_query = f"{state.query} comparação de melhores alternativas"
        # Query augmentation: adiciona termos para encontrar comparações
        
        search_results = self.firecrawl.search_companies(article_query, num_results=3)
        # Busca 3 artigos (suficiente para extração, não excessivo)

        # Extrai conteúdo dos artigos
        all_content = ""
        # Acumulador de conteúdo
        
        if hasattr(search_results, 'data') and search_results.data:
            # Defensive programming: verifica estrutura de dados
            
            for result in search_results.data:
                # Itera sobre resultados da busca
                
                url = result.get("url", "")
                # Safe access: get() com default vazio
                
                scraped = self.firecrawl.scrape_company_pages(url)
                # Scraping do conteúdo da página
                
                if scraped and hasattr(scraped, 'markdown'):
                    # Verifica se scraping foi bem-sucedido
                    
                    all_content += scraped.markdown[:1500] + "\n\n"
                    # Adiciona conteúdo limitado (1500 chars por artigo)
                    # Evita overflow de contexto

        # Usa LLM para extrair ferramentas
        messages = [
            SystemMessage(content=self.prompts.TOOL_EXTRACTION_SYSTEM),
            # System message: instrução base
            
            HumanMessage(content=self.prompts.tool_extraction_user(state.query, all_content))
            # Human message: prompt específico com dados
        ]

        try:
            response = self.llm.invoke(messages)
            # Chama LLM com mensagens
            
            tool_names = [
                name.strip()
                for name in response.content.strip().split("\n")
                if name.strip()
            ]
            # Parse da resposta:
            # 1. split("\n"): separa por linhas
            # 2. strip(): remove espaços
            # 3. if name.strip(): filtra linhas vazias
            
            print(f"Ferramentas encontradas: {', '.join(tool_names[:5])}")
            # Feedback visual: mostra ferramentas encontradas
            # [:5]: limita exibição a 5 primeiras
            
            return {"extracted_tools": tool_names}
            # Retorna update para o estado do workflow
            
        except Exception as e:
            print(f"Erro na extração: {e}")
            # Log de erro
            return {"extracted_tools": []}
            # Retorna lista vazia em caso de erro
            # Graceful degradation

    def _analyze_company_content(self, company_name: str, content: str) -> CompanyAnalysis:
        """Analisa conteúdo de empresa usando structured output"""
        # Método auxiliar para análise individual
        # Structured output: garante formato consistente
        
        structured_llm = self.llm.with_structured_output(CompanyAnalysis)
        # LangChain feature: força output em formato Pydantic
        # Benefício: validação automática, type safety

        messages = [
            SystemMessage(content=self.prompts.TOOL_ANALYSIS_SYSTEM),
            HumanMessage(content=self.prompts.tool_analysis_user(company_name, content))
        ]
        # Pattern consistente: system + human message

        try:
            analysis = structured_llm.invoke(messages)
            # LLM retorna objeto CompanyAnalysis válido
            return analysis
            
        except Exception as e:
            print(f"Erro na análise: {e}")
            # Log de erro para debugging
            
            return CompanyAnalysis(
                pricing_model="Desconhecido",
                is_open_source=None,
                tech_stack=[],
                description="Análise falhou",
                api_available=None,
                language_support=[],
                integration_capabilities=[]
            )
            # Fallback object: valores padrão em caso de erro
            # Permite workflow continuar mesmo com falhas parciais

    def _research_step(self, state: ResearchState) -> Dict[str, Any]:
        """Segundo passo: pesquisa detalhada de cada ferramenta"""
        # Step 2 do workflow: investigação aprofundada
        
        extracted_tools = getattr(state, "extracted_tools", [])
        # Safe access: getattr com default []

        # Fallback se não extraiu ferramentas
        if not extracted_tools:
            print("Fazendo busca direta...")
            # UX feedback: informa mudança de estratégia
            
            search_results = self.firecrawl.search_companies(state.query, num_results=4)
            # Busca direta com query original
            
            if hasattr(search_results, 'data') and search_results.data:
                tool_names = [
                    result.get("metadata", {}).get("title", result.get("title", "Unknown"))
                    for result in search_results.data
                ]
                # Extrai títulos dos resultados
                # Nested get(): acesso seguro a estrutura aninhada
                # Fallback chain: metadata.title → title → "Unknown"
            else:
                tool_names = []
                # Lista vazia se busca falhar
        else:
            tool_names = extracted_tools[:4]
            # Limita a 4 ferramentas para evitar sobrecarga

        print(f"🔬 Pesquisando: {', '.join(tool_names)}")
        # Feedback visual com emoji científico

        companies = []
        # Lista para acumular informações das empresas
        
        for tool_name in tool_names:
            # Itera sobre cada ferramenta para pesquisa individual
            
            # Busca site oficial
            tool_search_results = self.firecrawl.search_companies(
                tool_name + " site oficial", num_results=1
            )
            # Query específica: nome + "site oficial"
            # num_results=1: apenas resultado mais relevante

            if hasattr(tool_search_results, 'data') and tool_search_results.data:
                # Verifica se busca retornou resultados
                
                result = tool_search_results.data[0]
                # Pega primeiro (e único) resultado
                
                url = result.get("url", "")
                # Extrai URL do resultado

                company = CompanyInfo(
                    name=tool_name,
                    description=result.get("markdown", ""),
                    website=url
                )
                # Cria objeto CompanyInfo básico
                # Pydantic validation: garante tipos corretos

                # Scraping detalhado
                scraped = self.firecrawl.scrape_company_pages(url)
                # Extrai conteúdo completo da página
                
                if scraped and hasattr(scraped, 'markdown'):
                    # Verifica sucesso do scraping
                    
                    content = scraped.markdown
                    # Obtém conteúdo em markdown
                    
                    analysis = self._analyze_company_content(company.name, content)
                    # Análise usando LLM com structured output

                    # Atualiza informações
                    company.pricing_model = analysis.pricing_model
                    company.is_open_source = analysis.is_open_source
                    company.tech_stack = analysis.tech_stack
                    company.description = analysis.description
                    company.api_available = analysis.api_available
                    company.language_support = analysis.language_support
                    company.integration_capabilities = analysis.integration_capabilities
                    # Merge de dados: básicos + análise detalhada

                companies.append(company)
                # Adiciona empresa à lista de resultados

        return {"companies": companies}
        # Retorna lista de empresas para o estado

    def _analyze_step(self, state: ResearchState) -> Dict[str, Any]:
        """Terceiro passo: gera recomendações finais"""
        # Step 3: síntese e recomendações
        
        print("📊 Gerando recomendações...")
        # Feedback com emoji de análise

        # Serializa dados das empresas
        company_data = ", ".join([
            company.model_dump_json() for company in state.companies
        ])
        # Converte objetos Pydantic em JSON
        # model_dump_json(): serialização Pydantic
        # join(): concatena em string única

        messages = [
            SystemMessage(content=self.prompts.RECOMMENDATIONS_SYSTEM),
            HumanMessage(content=self.prompts.recommendations_user(state.query, company_data))
        ]
        # Pattern consistente para LLM

        try:
            response = self.llm.invoke(messages)
            # Gera recomendações usando LLM
            return {"analysis": response.content}
            # Retorna análise textual
            
        except Exception as e:
            print(f"Erro nas recomendações: {e}")
            return {"analysis": "Não foi possível gerar recomendações no momento."}
            # Fallback amigável em caso de erro

    async def process_query(self, query: str) -> str:
        """
        Processa query do usuário usando workflow estruturado
        
        Args:
            query: Consulta do usuário
            
        Returns:
            Resposta formatada com resultados e recomendações
        """
        # Método público principal: interface do agente
        
        try:
            # Executa workflow
            initial_state = ResearchState(query=query)
            # Cria estado inicial com query do usuário
            
            final_state = self.workflow.invoke(initial_state)
            # Executa workflow completo: extract → research → analyze
            # invoke(): método LangGraph para execução
            
            result = ResearchState(**final_state)
            # Reconstrói objeto tipado a partir do resultado

            # Formata resposta
            response_parts = []
            # Lista para construir resposta formatada
            
            response_parts.append(f"📋 **Resultados para: {query}**\n")
            # Header com query original

            if result.companies:
                # Se encontrou empresas
                
                response_parts.append("🏢 **Empresas/Ferramentas Encontradas:**\n")
                # Seção de empresas
                
                for i, company in enumerate(result.companies, 1):
                    # Enumera empresas (começa em 1)
                    
                    company_info = [
                        f"**{i}. {company.name}**",
                        f"🌐 Website: {company.website}",
                        f"💰 Preços: {company.pricing_model or 'N/A'}",
                        f"📖 Open Source: {'Sim' if company.is_open_source else 'Não' if company.is_open_source is False else 'N/A'}"
                    ]
                    # Informações básicas sempre presentes
                    # Ternary complex: trata 3 estados (True/False/None)

                    if company.tech_stack:
                        company_info.append(f"🛠️ Tecnologias: {', '.join(company.tech_stack[:3])}")
                    # Mostra tecnologias (máximo 3)

                    if company.language_support:
                        company_info.append(f"💻 Linguagens: {', '.join(company.language_support[:3])}")
                    # Mostra linguagens (máximo 3)

                    if company.api_available is not None:
                        api_status = "✅ Disponível" if company.api_available else "❌ Não disponível"
                        company_info.append(f"🔌 API: {api_status}")
                    # Status de API com emojis visuais

                    if company.description and company.description != "Análise falhou":
                        company_info.append(f"📝 Descrição: {company.description}")
                    # Descrição se disponível e válida

                    response_parts.append("\n".join(company_info) + "\n")
                    # Junta informações da empresa

            if result.analysis:
                response_parts.append(f"💡 **Recomendações:**\n{result.analysis}")
            # Adiciona recomendações se disponíveis

            return "\n".join(response_parts)
            # Junta todas as partes da resposta

        except Exception as e:
            return f"❌ Erro ao processar consulta: {str(e)}"
            # Mensagem de erro amigável

    def get_workflow_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o workflow"""
        # Método de introspecção: informações sobre capacidades
        
        return {
            "name": "Research Workflow Agent",
            "version": "1.0.0",
            "steps": ["extract_tools", "research", "analyze"],
            "capabilities": [
                "Extração de ferramentas de artigos",
                "Pesquisa detalhada de empresas",
                "Análise comparativa",
                "Recomendações técnicas"
            ]
        }
        # Metadados estruturados sobre o agente
        # Útil para debugging e documentação
