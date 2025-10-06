# Voice input feature

The robot control chat now supports voice input using the Web Speech API.

## How it works

The application uses the browser's built-in Web Speech API (Speech Recognition), which is available in most modern browsers. No additional backend API or API keys are required.

## Usage

1. Open the chat interface at http://localhost:8080
2. Click the microphone button (left side of the input box)
3. Allow microphone access when prompted by the browser
4. Speak your command clearly
5. The button turns red and pulses while listening
6. Click again to stop manually, or it will stop automatically when you finish speaking
7. The recognized text appears in the input box
8. Press Send or Enter to submit the command

## Browser compatibility

**Supported browsers:**
- Google Chrome (desktop and Android)
- Microsoft Edge
- Safari (desktop and iOS)
- Opera

**Not supported:**
- Firefox (does not support Web Speech API)
- Older browser versions

## Features

- **Visual feedback**: The microphone button turns red and pulses during recording
- **Automatic stop**: Recognition stops automatically after speech ends
- **Manual control**: Click the button again to stop recording manually
- **Error handling**: Clear messages for permission issues or unsupported browsers
- **Graceful degradation**: If not supported, the button is disabled with a tooltip explanation

## Technical details

**API**: Web Speech API (SpeechRecognition)
**Language**: English (en-US)
**Mode**: Single utterance (continuous = false)
**Results**: Final results only (interimResults = false)

## Privacy

The Web Speech API may send audio to cloud services depending on the browser implementation:
- Chrome/Edge: Audio is sent to Google's servers for processing
- Safari: Uses Apple's speech recognition services

Audio is not stored by this application. For privacy concerns, check your browser's privacy policy regarding speech recognition.

## Troubleshooting

**Button is disabled/grayed out**
- Use a supported browser (Chrome, Edge, or Safari)

**Microphone permission denied**
- Check browser address bar for permission prompts
- Go to browser settings → Site settings → Microphone
- Ensure microphone is allowed for localhost:8080

**No speech detected**
- Ensure your microphone is working (test in system settings)
- Speak clearly and close to the microphone
- Check browser has microphone access
- Try restarting the browser

**Recognition stops immediately**
- Background noise might be interfering
- Try in a quieter environment
- Ensure microphone input level is adequate

## Alternative: OpenAI Whisper API

If you prefer a more robust solution or need offline support, you can integrate OpenAI's Whisper API:

1. Add audio recording in the frontend
2. Send audio file to backend
3. Use OpenAI Whisper API to transcribe
4. Return transcription to frontend

This would require additional backend code and OpenAI API usage (costs apply).

