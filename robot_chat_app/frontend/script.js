let ws = null;
let isTyping = false;
let recognition = null;
let isRecording = false;

const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const voiceBtn = document.getElementById('voiceBtn');
const objectsList = document.getElementById('objectsList');
const robotStatus = document.getElementById('robotStatus');
const statusText = document.getElementById('statusText');
const refreshBtn = document.getElementById('refreshBtn');

function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('Connected to server');
        updateStatus(true);
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
    };
    
    ws.onclose = () => {
        console.log('Disconnected from server');
        updateStatus(false);
        setTimeout(connect, 3000);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateStatus(false);
    };
}

function updateStatus(connected) {
    if (connected) {
        robotStatus.className = 'status-indicator connected';
        statusText.textContent = 'Connected';
    } else {
        robotStatus.className = 'status-indicator disconnected';
        statusText.textContent = 'Disconnected';
    }
}

function handleMessage(data) {
    switch (data.type) {
        case 'system':
            addMessage('system', data.content);
            break;
        case 'user':
            addMessage('user', data.content);
            break;
        case 'assistant':
            removeTypingIndicator();
            addMessage('assistant', data.content);
            break;
        case 'error':
            removeTypingIndicator();
            addMessage('error', data.content);
            break;
        case 'typing':
            addTypingIndicator();
            break;
        case 'objects':
            updateObjectsList(data.objects);
            break;
    }
}

function addMessage(type, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addTypingIndicator() {
    if (isTyping) return;
    
    isTyping = true;
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant';
    typingDiv.id = 'typingIndicator';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'typing';
    contentDiv.textContent = 'Thinking...';
    
    typingDiv.appendChild(contentDiv);
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
        isTyping = false;
    }
}

function updateObjectsList(objects) {
    if (!objects || objects.length === 0) {
        objectsList.innerHTML = '<p class="no-objects">No objects detected</p>';
        return;
    }
    
    objectsList.innerHTML = '';
    objects.forEach(obj => {
        const item = document.createElement('div');
        item.className = 'object-item';
        
        const colorClass = `color-${obj.color_name}`;
        
        item.innerHTML = `
            <div class="object-name">
                <span class="color-badge ${colorClass}">${obj.color_name}</span>
                ${obj.class_name}
            </div>
            <div class="object-coords">Position: (${obj.x}, ${obj.y}, ${obj.z})</div>
        `;
        
        objectsList.appendChild(item);
    });
}

function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || !ws || ws.readyState !== WebSocket.OPEN) return;
    
    ws.send(JSON.stringify({
        type: 'message',
        content: message
    }));
    
    chatInput.value = '';
    chatInput.focus();
}

function refreshObjects() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    
    ws.send(JSON.stringify({
        type: 'refresh_objects'
    }));
}

sendBtn.addEventListener('click', sendMessage);

chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

refreshBtn.addEventListener('click', refreshObjects);

// Initialize speech recognition
function initSpeechRecognition() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        console.warn('Speech recognition not supported');
        voiceBtn.classList.add('disabled');
        voiceBtn.title = 'Speech recognition not supported in this browser';
        return;
    }
    
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    
    recognition.onstart = () => {
        isRecording = true;
        voiceBtn.classList.add('recording');
        voiceBtn.title = 'Listening... Click to stop';
        chatInput.placeholder = 'Listening...';
    };
    
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        chatInput.value = transcript;
        chatInput.focus();
    };
    
    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        stopRecording();
        
        if (event.error === 'not-allowed') {
            addMessage('system', 'Microphone access denied. Please allow microphone access in browser settings.');
        } else if (event.error !== 'no-speech' && event.error !== 'aborted') {
            addMessage('system', 'Speech recognition error: ' + event.error);
        }
    };
    
    recognition.onend = () => {
        stopRecording();
    };
}

function startRecording() {
    if (!recognition) {
        addMessage('system', 'Speech recognition not available in this browser. Try Chrome, Edge, or Safari.');
        return;
    }
    
    try {
        recognition.start();
    } catch (e) {
        console.error('Failed to start recognition:', e);
    }
}

function stopRecording() {
    isRecording = false;
    voiceBtn.classList.remove('recording');
    voiceBtn.title = 'Click to speak';
    chatInput.placeholder = 'Type a command or click microphone to speak';
}

function toggleVoiceRecording() {
    if (isRecording) {
        recognition.stop();
    } else {
        startRecording();
    }
}

voiceBtn.addEventListener('click', toggleVoiceRecording);

initSpeechRecognition();
connect();

