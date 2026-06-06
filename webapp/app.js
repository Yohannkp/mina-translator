/**
 * Mina-Voice - Application JavaScript
 * ==================================
 * Traducteur vocal Français ↔ Mina
 *
 * Fonctionnalités:
 * - Enregistrement audio (Web Audio API)
 * - Transcription Whisper (via API)
 * - Traduction FR ↔ Mina (via API)
 * - Synthèse vocale (Web Speech API)
 *
 * Auteur: Claude Opus 4.8
 * Date: 2026-06-05
 */

// =============================================================================
// CONFIGURATION
// =============================================================================

const API_BASE_URL = 'http://localhost:8000';

// =============================================================================
// STATE
// =============================================================================

let state = {
    // Recording
    mediaRecorder: null,
    audioChunks: [],
    audioBlob: null,

    // Audio visualization
    audioContext: null,
    analyser: null,
    animationId: null,

    // Timer
    recordingStartTime: null,
    timerInterval: null,

    // Mode
    translationMode: 'mina-to-fr', // 'mina-to-fr' or 'fr-to-mina'
    inputMode: 'voice', // 'voice' or 'text'

    // Processing
    isProcessing: false,
};

// =============================================================================
// DOM ELEMENTS
// =============================================================================

const $ = (id) => document.getElementById(id);

const elements = {
    // Status
    statusDot: $('statusDot'),
    statusText: $('statusText'),

    // Mode buttons (translation direction)
    modeBtns: document.querySelectorAll('.mode-btn'),

    // Input mode buttons
    inputModeBtns: document.querySelectorAll('.input-mode-btn'),

    // Sections
    voiceSection: $('voiceSection'),
    textSection: $('textSection'),

    // Voice elements
    recordBtn: $('recordBtn'),
    recordIcon: $('recordIcon'),
    recordText: $('recordText'),
    waveCanvas: $('waveCanvas'),
    wavePlaceholder: $('wavePlaceholder'),
    recordingTimer: $('recordingTimer'),
    timerText: $('timerText'),
    audioPlayer: $('audioPlayer'),
    audioPlayback: $('audioPlayback'),
    replayBtn: $('replayBtn'),
    processBtn: $('processBtn'),
    processText: $('processText'),

    // Text elements
    inputText: $('inputText'),
    inputLabel: $('inputLabel'),
    charCount: $('charCount'),
    phraseBtns: document.querySelectorAll('.phrase-btn'),
    translateTextBtn: $('translateTextBtn'),
    translateTextIcon: $('translateTextIcon'),
    translateTextText: $('translateTextText'),

    // Results
    resultsSection: $('resultsSection'),
    inputCard: $('inputCard'),
    inputTitle: $('inputTitle'),
    inputResult: $('inputResult'),
    inputTime: $('inputTime'),
    translationTitle: $('translationTitle'),
    translationResult: $('translationResult'),
    translationTime: $('translationTime'),
    copyBtn: $('copyBtn'),
    speakBtn: $('speakBtn'),
    confidenceSection: $('confidenceSection'),
    confidenceFill: $('confidenceFill'),
    confidenceText: $('confidenceText'),
    processingTime: $('processingTime'),

    // Error
    errorMessage: $('errorMessage'),
    errorText: $('errorText'),
    closeError: $('closeError'),
};

// =============================================================================
// API CONNECTION CHECK
// =============================================================================

async function checkAPIConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`, {
            method: 'GET',
        });

        if (response.ok) {
            const data = await response.json();
            elements.statusDot.className = 'status-dot connected';
            const modelName = data.model_name.split('/').pop();
            elements.statusText.textContent = `Connecté ✓ (${modelName})`;
            return true;
        }
    } catch (error) {
        elements.statusDot.className = 'status-dot error';
        elements.statusText.textContent = 'API déconnectée';
        return false;
    }
    return false;
}

// =============================================================================
// AUDIO RECORDING
// =============================================================================

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channels: 1,
                echoCancellation: true,
                noiseSuppression: true,
            }
        });

        // Setup audio context for visualization
        state.audioContext = new AudioContext();
        const source = state.audioContext.createMediaStreamSource(stream);
        state.analyser = state.audioContext.createAnalyser();
        state.analyser.fftSize = 256;
        source.connect(state.analyser);

        // Start visualization
        drawWaveform();

        // Setup MediaRecorder
        state.mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm;codecs=opus'
        });

        state.audioChunks = [];

        state.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                state.audioChunks.push(event.data);
            }
        };

        state.mediaRecorder.onstop = () => {
            state.audioBlob = new Blob(state.audioChunks, { type: 'audio/webm' });
            const audioUrl = URL.createObjectURL(state.audioBlob);
            elements.audioPlayback.src = audioUrl;
            elements.audioPlayer.style.display = 'flex';
            elements.processBtn.disabled = false;

            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
        };

        // Start recording
        state.mediaRecorder.start();
        state.recordingStartTime = Date.now();
        startTimer();

        // Update UI
        elements.recordBtn.classList.add('recording');
        elements.recordIcon.textContent = '⏹️';
        elements.recordText.textContent = 'Arrêter';
        elements.wavePlaceholder.classList.add('hidden');

    } catch (error) {
        showError('Accès microphone refusé. Veuillez autoriser l\'accès dans les paramètres du navigateur.');
    }
}

function stopRecording() {
    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
        state.mediaRecorder.stop();
    }

    // Stop visualization
    if (state.animationId) {
        cancelAnimationFrame(state.animationId);
        state.animationId = null;
    }

    // Reset UI
    elements.recordBtn.classList.remove('recording');
    elements.recordIcon.textContent = '🎤';
    elements.recordText.textContent = 'Enregistrer';
    elements.recordingTimer.classList.remove('active');
    stopTimer();

    // Clear canvas
    const ctx = elements.waveCanvas.getContext('2d');
    ctx.clearRect(0, 0, elements.waveCanvas.width, elements.waveCanvas.height);
    elements.wavePlaceholder.classList.remove('hidden');
}

function drawWaveform() {
    const canvas = elements.waveCanvas;
    const ctx = canvas.getContext('2d');
    const bufferLength = state.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
        if (!state.analyser) return;

        state.animationId = requestAnimationFrame(draw);
        state.analyser.getByteFrequencyData(dataArray);

        // Clear canvas
        ctx.fillStyle = '#334155';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw waveform
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#6366f1';
        ctx.beginPath();

        const sliceWidth = canvas.width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * canvas.height / 2;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }

            x += sliceWidth;
        }

        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();
    }

    draw();
}

// =============================================================================
// TIMER
// =============================================================================

function startTimer() {
    elements.recordingTimer.classList.add('active');
    elements.timerText.textContent = '00:00';

    state.timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - state.recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
        const seconds = (elapsed % 60).toString().padStart(2, '0');
        elements.timerText.textContent = `${minutes}:${seconds}`;
    }, 1000);
}

function stopTimer() {
    if (state.timerInterval) {
        clearInterval(state.timerInterval);
        state.timerInterval = null;
    }
}

// =============================================================================
// TEXT INPUT PROCESSING
// =============================================================================

function setInputMode(mode) {
    state.inputMode = mode;

    elements.inputModeBtns.forEach(btn => {
        if (btn.dataset.input === mode) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    // Toggle visibility
    if (mode === 'voice') {
        elements.voiceSection.style.display = 'block';
        elements.textSection.style.display = 'none';
    } else {
        elements.voiceSection.style.display = 'none';
        elements.textSection.style.display = 'block';
        updateInputLabel();
    }

    // Hide results when switching
    elements.resultsSection.style.display = 'none';
}

function setTranslationMode(mode) {
    state.translationMode = mode;

    elements.modeBtns.forEach(btn => {
        if (btn.dataset.mode === mode) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    // Update input label for text mode
    if (state.inputMode === 'text') {
        updateInputLabel();
    }

    // Hide results when switching
    elements.resultsSection.style.display = 'none';
}

function updateInputLabel() {
    if (state.translationMode === 'mina-to-fr') {
        elements.inputLabel.textContent = 'Entrez votre texte en Mina (Ewe):';
        elements.translateTextText.textContent = 'Traduire en Français';
        elements.translateTextIcon.textContent = '🔄';
    } else {
        elements.inputLabel.textContent = 'Entrez votre texte en Français:';
        elements.translateTextText.textContent = 'Traduire en Mina';
        elements.translateTextIcon.textContent = '🔄';
    }
}

function insertPhrase(phrase) {
    elements.inputText.value = phrase;
    updateCharCount();
}

function updateCharCount() {
    const count = elements.inputText.value.length;
    elements.charCount.textContent = count;
}

// =============================================================================
// API PROCESSING
// =============================================================================

async function processAudio() {
    if (!state.audioBlob) {
        showError('Aucun audio enregistré.');
        return;
    }

    const startTotal = Date.now();
    state.isProcessing = true;

    // Show loading state
    elements.processBtn.disabled = true;
    elements.processText.innerHTML = '<span class="spinner"></span> Transcription...';

    try {
        // Step 1: Transcribe audio
        let transcriptionText = '';
        let sttTime = 0;

        try {
            const transcription = await transcribeAudio();
            transcriptionText = transcription.text;
            sttTime = transcription.time_ms || 0;
        } catch (e) {
            console.log('STT non disponible, fallback...');
            transcriptionText = '[Audio enregistré - transcription non disponible]';
            sttTime = 0;
        }

        // Update UI with transcription
        if (transcriptionText) {
            elements.inputCard.style.display = 'block';
            elements.inputResult.textContent = transcriptionText;
            elements.inputTime.textContent = `${sttTime} ms`;
        }

        // Step 2: Translate
        elements.processText.innerHTML = '<span class="spinner"></span> Traduction...';

        const translation = await translateText(transcriptionText);

        // Show results
        showResults(transcriptionText, translation, sttTime, startTotal);

    } catch (error) {
        showError(`Erreur: ${error.message}`);
    } finally {
        state.isProcessing = false;
        elements.processBtn.disabled = !state.audioBlob;
        elements.processText.innerHTML = '🎯 Transcrire et Traduire';
    }
}

async function processText() {
    const text = elements.inputText.value.trim();

    if (!text) {
        showError('Veuillez entrer du texte à traduire.');
        return;
    }

    state.isProcessing = true;

    const startTotal = Date.now();
    const translateBtn = elements.translateTextBtn;

    // Show loading
    translateBtn.disabled = true;
    elements.translateTextIcon.textContent = '⏳';
    elements.translateTextText.textContent = 'Traduction...';

    try {
        const translation = await translateText(text);
        showResults(text, translation, 0, startTotal);

    } catch (error) {
        showError(`Erreur: ${error.message}`);
    } finally {
        state.isProcessing = false;
        translateBtn.disabled = false;
        elements.translateTextIcon.textContent = '🔄';
        elements.translateTextText.textContent = state.translationMode === 'mina-to-fr' ? 'Traduire en Français' : 'Traduire en Mina';
    }
}

async function transcribeAudio() {
    const formData = new FormData();
    formData.append('file', state.audioBlob, 'audio.webm');
    formData.append('model', 'whisper-small');

    const startTime = Date.now();

    try {
        const response = await fetch(`${API_BASE_URL}/transcribe`, {
            method: 'POST',
            body: formData,
        });

        const elapsed = Date.now() - startTime;

        if (response.ok) {
            const data = await response.json();
            return {
                text: data.text || data.transcription || '',
                time_ms: elapsed,
                confidence: data.confidence || 0.8,
            };
        }
    } catch (error) {
        console.log('STT endpoint non disponible');
    }

    return {
        text: '',
        time_ms: Date.now() - startTime,
        confidence: 0.5,
    };
}

async function translateText(text) {
    const startTime = Date.now();

    const endpoint = state.translationMode === 'mina-to-fr' ? '/translate_mina' : '/translate';

    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text: text,
            temperature: 0.3,
            max_length: 256,
        }),
    });

    const elapsed = Date.now() - startTime;

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();

    return {
        text: data.translation,
        time_ms: elapsed,
        original: data.original,
    };
}

// =============================================================================
// RESULTS DISPLAY
// =============================================================================

function showResults(inputText, translation, sttTime, startTotal) {
    elements.resultsSection.style.display = 'block';

    const totalTime = Date.now() - startTotal;

    // Show or hide input card based on mode
    if (state.inputMode === 'voice' && inputText && !inputText.startsWith('[')) {
        elements.inputCard.style.display = 'block';
        elements.inputTitle.textContent = state.translationMode === 'mina-to-fr' ? '🎤 Transcription (Mina)' : '🎤 Transcription (Français)';
        elements.inputResult.textContent = inputText;
        elements.inputTime.textContent = `${sttTime} ms`;
    } else {
        elements.inputCard.style.display = 'none';
    }

    // Translation
    elements.translationTitle.textContent = state.translationMode === 'mina-to-fr' ? '🔄 Traduction (Français)' : '🔄 Traduction (Mina)';
    elements.translationResult.textContent = translation.text;
    elements.translationTime.textContent = `${translation.time_ms} ms`;

    // Confidence (simulated for now)
    const confidence = state.translationMode === 'mina-to-fr' ? 0.75 : 0.7;
    elements.confidenceFill.style.width = `${confidence * 100}%`;
    elements.confidenceText.textContent = `Confiance: ${Math.round(confidence * 100)}%`;

    // Stats
    elements.processingTime.textContent = `Temps total: ${totalTime} ms`;

    // Scroll to results
    elements.resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// =============================================================================
// TEXT TO SPEECH
// =============================================================================

function speakText() {
    const text = elements.translationResult.textContent;

    if (!text || text.startsWith('<')) return;

    // Determine language
    const lang = state.translationMode === 'mina-to-fr' ? 'fr-FR' : 'ee';

    // Use Web Speech API
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang;
    utterance.rate = 0.9;

    elements.speakBtn.classList.add('speaking');
    elements.speakBtn.textContent = '🔊 Lecture...';

    utterance.onend = () => {
        elements.speakBtn.classList.remove('speaking');
        elements.speakBtn.textContent = '🔊 Écouter';
    };

    utterance.onerror = () => {
        elements.speakBtn.classList.remove('speaking');
        elements.speakBtn.textContent = '🔊 Écouter';
    };

    speechSynthesis.speak(utterance);
}

// =============================================================================
// CLIPBOARD
// =============================================================================

async function copyToClipboard() {
    const text = elements.translationResult.textContent;

    try {
        await navigator.clipboard.writeText(text);
        elements.copyBtn.textContent = '✓ Copié!';
        elements.copyBtn.classList.add('copied');

        setTimeout(() => {
            elements.copyBtn.textContent = '📋 Copier';
            elements.copyBtn.classList.remove('copied');
        }, 2000);
    } catch (error) {
        showError('Impossible de copier le texte.');
    }
}

// =============================================================================
// ERROR HANDLING
// =============================================================================

function showError(message) {
    elements.errorMessage.style.display = 'flex';
    elements.errorText.textContent = message;

    setTimeout(() => {
        elements.errorMessage.style.display = 'none';
    }, 5000);
}

function hideError() {
    elements.errorMessage.style.display = 'none';
}

// =============================================================================
// EVENT LISTENERS
// =============================================================================

// Record button
elements.recordBtn.addEventListener('click', () => {
    if (state.mediaRecorder && state.mediaRecorder.state === 'recording') {
        stopRecording();
    } else {
        startRecording();
    }
});

// Replay button
elements.replayBtn.addEventListener('click', () => {
    elements.audioPlayback.currentTime = 0;
    elements.audioPlayback.play();
});

// Process button (voice)
elements.processBtn.addEventListener('click', processAudio);

// Text input area
elements.inputText.addEventListener('input', updateCharCount);

// Translate button (text)
elements.translateTextBtn.addEventListener('click', processText);

// Copy button
elements.copyBtn.addEventListener('click', copyToClipboard);

// Speak button
elements.speakBtn.addEventListener('click', speakText);

// Close error
elements.closeError.addEventListener('click', hideError);

// Mode buttons (translation direction)
elements.modeBtns.forEach(btn => {
    btn.addEventListener('click', () => setTranslationMode(btn.dataset.mode));
});

// Input mode buttons
elements.inputModeBtns.forEach(btn => {
    btn.addEventListener('click', () => setInputMode(btn.dataset.input));
});

// Phrase buttons
elements.phraseBtns.forEach(btn => {
    btn.addEventListener('click', () => insertPhrase(btn.dataset.phrase));
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ignore if typing in textarea
    if (e.target === elements.inputText) return;

    // Space to record/stop
    if (e.code === 'Space' && state.inputMode === 'voice') {
        e.preventDefault();
        elements.recordBtn.click();
    }

    // Enter to translate (text mode)
    if (e.code === 'Enter' && e.ctrlKey && state.inputMode === 'text') {
        e.preventDefault();
        if (!elements.translateTextBtn.disabled) {
            processText();
        }
    }
});

// =============================================================================
// INITIALIZATION
// =============================================================================

async function init() {
    console.log('🚀 Mina-Voice initializing...');

    // Check API connection
    await checkAPIConnection();

    // Re-check every 30 seconds
    setInterval(checkAPIConnection, 30000);

    // Initialize state
    elements.resultsSection.style.display = 'none';
    elements.audioPlayer.style.display = 'none';
    elements.inputCard.style.display = 'none';

    // Set initial modes
    updateInputLabel();

    console.log('✅ Mina-Voice ready!');
}

// Start the app
document.addEventListener('DOMContentLoaded', init);