// Estado global da aplicação
const AppState = {
    selectedAgent: null,
    isLoading: false,
    messageHistory: []
};

// Elementos DOM
const Elements = {
    agentButtons: document.querySelectorAll('.agent-btn'),
    selectedAgentDiv: document.getElementById('selected-agent'),
    chatInput: document.getElementById('chat-input'),
    sendBtn: document.getElementById('send-btn'),
    chatMessages: document.getElementById('chat-messages'),
    loadingOverlay: document.getElementById('loading-overlay'),
    loadingText: document.getElementById('loading-text'),
    agentIndicator: document.getElementById('agent-indicator'),
    charCount: document.querySelector('.char-count'),
    errorModal: document.getElementById('error-modal'),
    errorMessage: document.getElementById('error-message'),
    closeModal: document.getElementById('close-modal'),
    errorOkBtn: document.getElementById('error-ok-btn')
};

// Configurações
const CONFIG = {
    maxMessageLength: 1000,
    apiEndpoint: '/chat',
    streamEndpoint: '/chat/stream'
};

// Inicialização da aplicação
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    setupEventListeners();
    updateUI();
    
    // Remove mensagem de boas-vindas se houver histórico
    if (AppState.messageHistory.length > 0) {
        clearWelcomeMessage();
    }
}

function setupEventListeners() {
    // Seleção de agentes
    Elements.agentButtons.forEach(btn => {
        btn.addEventListener('click', () => selectAgent(btn.dataset.agent));
    });
    
    // Envio de mensagens
    Elements.sendBtn.addEventListener('click', sendMessage);
    Elements.chatInput.addEventListener('keypress', handleKeyPress);
    Elements.chatInput.addEventListener('input', updateCharCount);
    
    // Modal de erro
    Elements.closeModal.addEventListener('click', closeErrorModal);
    Elements.errorOkBtn.addEventListener('click', closeErrorModal);
    
    // Fechar modal clicando fora
    Elements.errorModal.addEventListener('click', (e) => {
        if (e.target === Elements.errorModal) {
            closeErrorModal();
        }
    });
    
    // Escape key para fechar modal
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeErrorModal();
        }
    });
}

function selectAgent(agentType) {
    AppState.selectedAgent = agentType;
    
    // Atualizar visual dos botões
    Elements.agentButtons.forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.agent === agentType) {
            btn.classList.add('active');
        }
    });
    
    // Atualizar indicador
    const agentNames = {
        'mcp': 'Agente MCP',
        'workflow': 'Agente Workflow'
    };
    
    Elements.selectedAgentDiv.innerHTML = `
        <i class="fas fa-check-circle"></i>
        <span>Agente selecionado: ${agentNames[agentType]}</span>
    `;
    Elements.selectedAgentDiv.classList.add('active');
    
    Elements.agentIndicator.textContent = agentNames[agentType];
    
    updateUI();
    
    // Focar no input após seleção
    Elements.chatInput.focus();
}

function updateUI() {
    const hasAgent = AppState.selectedAgent !== null;
    const hasText = Elements.chatInput.value.trim().length > 0;
    
    Elements.chatInput.disabled = !hasAgent || AppState.isLoading;
    Elements.sendBtn.disabled = !hasAgent || !hasText || AppState.isLoading;
    
    if (hasAgent) {
        Elements.chatInput.placeholder = "Digite sua pergunta aqui...";
    } else {
        Elements.chatInput.placeholder = "Selecione um agente primeiro";
    }
}

function handleKeyPress(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function updateCharCount() {
    const length = Elements.chatInput.value.length;
    Elements.charCount.textContent = `${length}/${CONFIG.maxMessageLength}`;
    
    if (length > CONFIG.maxMessageLength * 0.9) {
        Elements.charCount.style.color = '#dc2626';
    } else {
        Elements.charCount.style.color = '#6b7280';
    }
    
    updateUI();
}

async function sendMessage() {
    const message = Elements.chatInput.value.trim();
    
    if (!message || !AppState.selectedAgent || AppState.isLoading) {
        return;
    }
    
    // Adicionar mensagem do usuário
    addMessage(message, 'user');
    
    // Limpar input
    Elements.chatInput.value = '';
    updateCharCount();
    
    // Mostrar loading
    showLoading('Processando sua solicitação...');
    
    try {
        // Enviar para API
        const response = await fetch(CONFIG.apiEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                agent_type: AppState.selectedAgent
            })
        });
        
        if (!response.ok) {
            throw new Error(`Erro ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.status === 'error') {
            throw new Error(data.response);
        }
        
        // Adicionar resposta do bot
        addMessage(data.response, 'bot', data.agent_type);
        
    } catch (error) {
        console.error('Erro ao enviar mensagem:', error);
        showError(`Erro ao processar mensagem: ${error.message}`);
        
        // Adicionar mensagem de erro no chat
        addMessage(
            '❌ Desculpe, ocorreu um erro ao processar sua mensagem. Tente novamente.', 
            'bot', 
            'error'
        );
    } finally {
        hideLoading();
        updateUI();
        Elements.chatInput.focus();
    }
}

function addMessage(content, sender, agentType = null) {
    // Limpar mensagem de boas-vindas na primeira mensagem
    if (AppState.messageHistory.length === 0) {
        clearWelcomeMessage();
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    // Processar conteúdo (converter quebras de linha, etc.)
    const processedContent = processMessageContent(content);
    messageContent.innerHTML = processedContent;
    
    const messageTime = document.createElement('div');
    messageTime.className = 'message-time';
    messageTime.textContent = new Date().toLocaleTimeString('pt-BR', {
        hour: '2-digit',
        minute: '2-digit'
    });
    
    messageDiv.appendChild(messageContent);
    messageContent.appendChild(messageTime);
    
    Elements.chatMessages.appendChild(messageDiv);
    
    // Scroll para a última mensagem
    Elements.chatMessages.scrollTop = Elements.chatMessages.scrollHeight;
    
    // Adicionar ao histórico
    AppState.messageHistory.push({
        content,
        sender,
        agentType,
        timestamp: new Date()
    });
}

function processMessageContent(content) {
    // Converter quebras de linha em <br>
    let processed = content.replace(/\n/g, '<br>');
    
    // Converter URLs em links
    const urlRegex = /(https?:\/\/[^\s]+)/g;
    processed = processed.replace(urlRegex, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
    
    // Processar formatação básica de markdown
    processed = processed.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    processed = processed.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    return processed;
}

function clearWelcomeMessage() {
    const welcomeMessage = Elements.chatMessages.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }
}

function showLoading(text = 'Carregando...') {
    AppState.isLoading = true;
    Elements.loadingText.textContent = text;
    Elements.loadingOverlay.classList.add('active');
    updateUI();
}

function hideLoading() {
    AppState.isLoading = false;
    Elements.loadingOverlay.classList.remove('active');
    updateUI();
}

function showError(message) {
    Elements.errorMessage.textContent = message;
    Elements.errorModal.classList.add('active');
}

function closeErrorModal() {
    Elements.errorModal.classList.remove('active');
}

// Utility functions
function formatTime(date) {
    return date.toLocaleTimeString('pt-BR', {
        hour: '2-digit',
        minute: '2-digit'
    });
}

function formatDate(date) {
    return date.toLocaleDateString('pt-BR');
}

// Função para exportar histórico de chat (feature extra)
function exportChatHistory() {
    if (AppState.messageHistory.length === 0) {
        showError('Não há mensagens para exportar.');
        return;
    }
    
    const chatData = {
        exportDate: new Date().toISOString(),
        agent: AppState.selectedAgent,
        messages: AppState.messageHistory
    };
    
    const dataStr = JSON.stringify(chatData, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `chat-history-${formatDate(new Date())}.json`;
    link.click();
}

// Função para limpar chat (feature extra)
function clearChat() {
    if (confirm('Tem certeza que deseja limpar todo o histórico de chat?')) {
        AppState.messageHistory = [];
        Elements.chatMessages.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-content">
                    <i class="fas fa-hand-wave"></i>
                    <h3>Bem-vindo aos Agentes de IA!</h3>
                    <p>Escolha um agente acima e comece a conversar. Posso ajudar com:</p>
                    <ul>
                        <li><i class="fas fa-search"></i> Pesquisa de produtos e serviços</li>
                        <li><i class="fas fa-chart-line"></i> Análise comparativa de preços</li>
                        <li><i class="fas fa-lightbulb"></i> Recomendações técnicas</li>
                        <li><i class="fas fa-globe"></i> Web scraping inteligente</li>
                    </ul>
                </div>
            </div>
        `;
    }
}

// Adicionar atalhos de teclado (feature extra)
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter para enviar mensagem
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        sendMessage();
    }
    
    // Ctrl/Cmd + K para limpar chat
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        clearChat();
    }
    
    // Ctrl/Cmd + S para exportar
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        exportChatHistory();
    }
});

// Verificar status dos agentes ao carregar
async function checkAgentStatus() {
    try {
        const response = await fetch('/agents/info');
        const data = await response.json();
        
        // Atualizar disponibilidade dos agentes na UI
        data.agents.forEach(agent => {
            const btn = document.querySelector(`[data-agent="${agent.type}"]`);
            if (btn && !agent.available) {
                btn.style.opacity = '0.5';
                btn.style.cursor = 'not-allowed';
                btn.title = 'Agente não disponível - Verifique configurações';
            }
        });
        
    } catch (error) {
        console.warn('Não foi possível verificar status dos agentes:', error);
    }
}

// Verificar status na inicialização
checkAgentStatus();

// Feature de reconexão automática em caso de erro de rede
let reconnectAttempts = 0;
const maxReconnectAttempts = 3;

async function attemptReconnect() {
    if (reconnectAttempts >= maxReconnectAttempts) {
        showError('Não foi possível conectar ao servidor. Recarregue a página.');
        return;
    }
    
    reconnectAttempts++;
    showLoading(`Tentando reconectar... (${reconnectAttempts}/${maxReconnectAttempts})`);
    
    try {
        const response = await fetch('/health');
        if (response.ok) {
            hideLoading();
            reconnectAttempts = 0;
        } else {
            throw new Error('Servidor não responde');
        }
    } catch (error) {
        setTimeout(attemptReconnect, 2000);
    }
}

// Detectar perda de conexão
window.addEventListener('online', () => {
    if (reconnectAttempts > 0) {
        attemptReconnect();
    }
});

window.addEventListener('offline', () => {
    showError('Conexão perdida. Verifique sua internet.');
});